#include "clsolver.h"
#include <iomanip>
#include <ios>
#include <iostream>
#include <iterator>
#include <string>
#include <fstream>
#include <streambuf>
#include <sstream>
#include <queue>
#include <list>
#include <thread>
#include "presolver.h"
#include "cpusolver.h"

ClSolver::ClSolver()
{
    cl_int err = 0;
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if(platforms.empty()) {
        std::cout << "No OpenCL platforms found" << std::endl;
        return;
    }

    cl::Platform platform = platforms[0];

    std::cout << "Platform name: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
    std::cout << "Platform version: " << platform.getInfo<CL_PLATFORM_VERSION>() << std::endl;

    std::vector<cl::Device> devices;

    err = platform.getDevices(CL_DEVICE_TYPE_GPU|CL_DEVICE_TYPE_CPU, &devices);
    if(err != CL_SUCCESS) {
        std::cout << "getDevices failed" << std::endl;
        return;
    }

    // find available devices
    std::vector<cl::Device> availableDevices;
    for(auto dev: devices) {
        if (dev.getInfo<CL_DEVICE_AVAILABLE>(&err)) {
            if(err != CL_SUCCESS) {
                std::cout << "getInfo<CL_DEVICE_AVAILABLE> failed" << std::endl;
                return;
            }
            availableDevices.push_back(dev);
        }
    }

    if(availableDevices.empty()) {
        std::cout << "No devices found" << std::endl;
        return;
    }

    // select first available device
    device = availableDevices[0];

    std::cout << "selected Device: " << device.getInfo<CL_DEVICE_NAME>(&err) << std::endl;
    if(err != CL_SUCCESS) {
        std::cout << "getInfo<CL_DEVICE_NAME> failed" << std::endl;
        return;
    }

    context = cl::Context(device, nullptr, nullptr, nullptr, &err);
    if(err != CL_SUCCESS) {
        std::cout << "cl::Context failed" << std::endl;
        return;
    }

    // load source code
    std::ifstream sourcefile("clqueens_pre.cl");
    std::string sourceStr((std::istreambuf_iterator<char>(sourcefile)),
                     std::istreambuf_iterator<char>());

    // create OpenCL program
    program = cl::Program(context, sourceStr, false, &err);
    if(err != CL_SUCCESS) {
        std::cout << "cl::Program failed" << std::endl;
        return;
    }
}

constexpr uint_fast8_t MINN = 2;
constexpr uint_fast8_t MAXN = 29;

constexpr size_t N_STACKS = 256; // number of stacks
constexpr size_t WORKGROUP_SIZE = 256;   // number of threads that are run in parallel
constexpr size_t STACK_SIZE = 512;      // number of elements in a stack

/*
 * GPU_DEPTH defines how many rows should be left for the GPU to solve,
 * the previous ones have to be solved with the cpu.
 * With a too high GPU_DEPTH, solving a board takes too long and the
 * GPU is detected as "hung" by the driver and reset or the system crashes.
 */
constexpr uint_fast8_t GPU_DEPTH = 12;

bool ClSolver::init(uint8_t boardsize, uint8_t placed)
{
    if(boardsize > MAXN || boardsize < MINN) {
        std::cout << "Invalid boardsize for ClSolver" << std::endl;
        return false;
    }

    if(placed >= boardsize) {
        std::cout << "Invalid number of placed queens for ClSolver" << std::endl;
        return false;
    }

    this->boardsize = boardsize;
    this->placed = placed;
    uint_fast8_t gpu_depth = GPU_DEPTH;
    if((boardsize - placed) < GPU_DEPTH) {
        gpu_depth = boardsize - placed;
        presolve_depth = 0;
    } else {
        presolve_depth = boardsize - placed - GPU_DEPTH;
    }

    std::ostringstream optionsStream;
    optionsStream << "-D N=" << std::to_string(boardsize)
                  << " -D PLACED=" << std::to_string(boardsize - gpu_depth)
                  << " -D N_STACKS=" << std::to_string(N_STACKS)
                  << " -D WORKGROUP_SIZE=" << std::to_string(WORKGROUP_SIZE)
                  << " -D STACK_SIZE=" << std::to_string(STACK_SIZE);

    std::string options = optionsStream.str();

    std::cout << "OpenCL Kernel Options: " << options << std::endl;

    cl_int builderr = program.build(options.c_str());
    if(builderr != CL_SUCCESS) {
        std::cout << "program.build failed: " << builderr << std::endl;
        cl_int err = 0;
        std::cout << "OpenCL build log:" << std::endl;
        auto buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device, &err);
        std::cout << buildlog << std::endl;
        if(err != CL_SUCCESS) {
            std::cout << "getBuildInfo<CL_PROGRAM_BUILD_LOG> failed" << std::endl;
        }
        return false;
    }

    return true;
}

typedef cl_uint result_type;

constexpr size_t NUM_BATCHES = 4;
constexpr size_t NUM_CMDQUEUES = 2;

typedef enum {FIRST, MID, LAST1, LAST2} STAGE_TYPE;

typedef struct {
    cl::Kernel clKernel;
    cl::Buffer clStageBuf;  // buffer for the data alive after this stage
    cl::Buffer clFillCount; // buffer that holds the fill status
    cl::Buffer clSum;       // buffer for the sum at the last stage
    cl::Event clStageDone;  // event when this stage is complete
    std::vector<cl_int> hostFillCount;
    STAGE_TYPE type;
    uint8_t stageIdx;
    uint8_t queens_start;   // number of queens at the beginning of this stage
    uint32_t expansion;     // maximum number of solutions generated from one input solution
} sieve_stage;

typedef struct {
    cl_uint bufferIdx;     // must be 32 Bit, to match OpenCL kernel
    uint32_t stageIdx;
} stage_work_item;

constexpr size_t PLACED_PER_STAGE = 2;  // number of queens placed per sieve stage

uint64_t ClSolver::solve_subboard(const std::vector<start_condition> &start)
{
    cl_int err = CL_SUCCESS;

    if(start.empty()) {
        return 0;
    }

    auto startIt = start.begin();

    // init presolver
    PreSolver pre(boardsize, placed, presolve_depth, *startIt);
    startIt++;

    cl::CommandQueue queue;
    // Create command queue.
    queue = cl::CommandQueue(context, device, 0, &err);
    if(err != CL_SUCCESS) {
        std::cout << "failed to create command queue: " << err << std::endl;
    }

    std::vector<sieve_stage> stages;
    uint8_t queens_left = boardsize - placed - presolve_depth; // number of queens to place till board is full
    // we need at least 2 (first stage) + 2 (mid stage) + 1 (last1 stage) queens left
    if(queens_left < 5) {
        std::cout << "not enough queens left" << std::endl;
        return 0;
    }

    // create first sieve stage
    sieve_stage first{};
    first.type = FIRST;
    first.stageIdx = 0;
    first.queens_start = queens_left;
    stages.push_back(first);
    queens_left -= 2;

    uint8_t cnt = 1;

    // create middle sieve stages
    while(queens_left > 2) {
        sieve_stage mid{};
        mid.type = MID;
        mid.stageIdx = cnt;
        mid.queens_start = queens_left;
        cnt++;
        stages.push_back(mid);
        queens_left -= 2;
    }

    // create last sieve stage
    sieve_stage last{};
    last.queens_start = queens_left;
    if(queens_left == 1) {
        last.type = LAST1;
    } else if (queens_left == 2) {
        last.type = LAST2;
    } else {
        std::cout << "wrong number of queens left" << std::endl;
        return 0;
    }

    last.stageIdx = cnt;
    stages.push_back(last);

    uint32_t max_expansion = 0;
    // compute max expansion factors
    for(auto& stage : stages) {
        uint32_t expansion = 1;
        for(uint32_t i = 0; i < PLACED_PER_STAGE; i++) {
            expansion *= boardsize - stage.queens_start - 1 - i;
        }
        stage.expansion = expansion;
        max_expansion = std::max(expansion, max_expansion);
    }
    std::cout << "Number of stages: " << stages.size() << std::endl;
    std::cout << "Maximum expansion: " << max_expansion << std::endl;

    // initialize OpenCL stuff
    for(size_t i = 0; i < stages.size(); i++) {
        sieve_stage& stage = stages.at(i);
        // Initialize stage buffer
        if(stage.type != LAST1 && stage.type != LAST2) {
            stage.clStageBuf = cl::Buffer(context, CL_MEM_READ_WRITE,
                                          N_STACKS * STACK_SIZE * sizeof(start_condition), nullptr, &err);
            if(err != CL_SUCCESS) {
                std::cout << "cl::Buffer clStageBuf failed: " << err << std::endl;
            }

            // host fill count buffer
            stage.hostFillCount = std::vector<cl_int>(N_STACKS, 0);

            // Initialize stage buffer element count to zero
            stage.clFillCount = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                          N_STACKS * sizeof(cl_int), stage.hostFillCount.data(), &err);
            if(err != CL_SUCCESS) {
                std::cout << "cl::Buffer clFillCount failed: " << err << std::endl;
            }
        }

        // Initialize output buffer
        if(stage.type == LAST1 || stage.type == LAST2) {
            stage.clSum = cl::Buffer(context, CL_MEM_READ_WRITE,
                                          N_STACKS * sizeof(result_type), nullptr, &err);
            if(err != CL_SUCCESS) {
                std::cout << "cl::Buffer clSum failed: " << err << std::endl;
            }

        }

        const auto prev_stage_idx = i - 1;
        // create device kernel
        switch (stage.type) {
            case FIRST:
                stage.clKernel = cl::Kernel(program, "first_step", &err);
                // input buffer
                // arg 0 set later
                // output buffer
                stage.clKernel.setArg(1, stage.clStageBuf);
                // output buffer fill status
                stage.clKernel.setArg(2, stage.clFillCount);
            break;
            case MID:
                stage.clKernel = cl::Kernel(program, "inter_step", &err);
                // input buffer
                stage.clKernel.setArg(0, stages.at(prev_stage_idx).clStageBuf);
                // offset
                // arg 1 set later
                // output buffer
                stage.clKernel.setArg(2, stage.clStageBuf);
                // output buffer fill status
                stage.clKernel.setArg(3, stage.clFillCount);
                // input buffer fill status
                stage.clKernel.setArg(4, stages.at(prev_stage_idx).clFillCount);
                break;
            case LAST1:
            case LAST2:
                stage.clKernel = cl::Kernel(program, "final_step", &err);
                // input buffer
                stage.clKernel.setArg(0, stages.at(prev_stage_idx).clStageBuf);
                // offset
                // arg 1 set later
                // input buffer fill status
                stage.clKernel.setArg(2, stages.at(prev_stage_idx).clFillCount);
                // output sum buffer
                stage.clKernel.setArg(3, stage.clSum);
                break;
            default:
                std::cout << "unexpected stage" << std::endl;
        }

        if(err != CL_SUCCESS) {
            std::cout << "cl::Kernel failed: " << err << std::endl;
        }

    }

    // TODO(sudden6): calculate this based on expansion and per stage
    const cl_int BUF_THRESHOLD = STACK_SIZE - max_expansion - 10;

    uint64_t result = 0;

    std::list<stage_work_item> work_queue{};
    cl::Buffer clInputBuf;
    std::vector<start_condition> hostStartBuf{N_STACKS};
    std::vector<cl_uint> hostOutputBuf(N_STACKS, 0);     // store the result of the last stage

    bool done = false;

    while (!done) {
        queue.finish();
        // fill step
        if(work_queue.empty() && !pre.empty()) {
            // insert new material at first sieve stage
            auto& stage = stages.at(0);
            // fill buffer
            auto hostBufIt = hostStartBuf.begin();
            while(hostBufIt != hostStartBuf.end()) {
                hostBufIt = pre.getNext(hostBufIt, hostStartBuf.cend());
                if(pre.empty() && (startIt == start.end())) {
                    // out of start conditions
                    break;
                }
                if(pre.empty()) {
                    pre = PreSolver(boardsize, placed, presolve_depth, *startIt);
                    startIt++;
                }
            }

            ssize_t dist = std::distance(hostStartBuf.begin(), hostBufIt);
            if(dist <= 0) {
                std::cout << "Error negative distance" << std::endl;
            }

            size_t input_cnt = static_cast<size_t>(dist);

            // upload input data to device
            clInputBuf = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                    input_cnt * sizeof(start_condition), hostStartBuf.data(), &err);
            if(err != CL_SUCCESS) {
                std::cout << "cl::Buffer clInputBuf failed: " << err << std::endl;
            }

            // input buffer
            stage.clKernel.setArg(0, clInputBuf);

            // Launch kernel on the compute device.
            err = queue.enqueueNDRangeKernel(stage.clKernel, cl::NullRange,
                                             cl::NDRange{input_cnt}, cl::NDRange{std::min(WORKGROUP_SIZE, N_STACKS)},
                                             nullptr, &stage.clStageDone);

            if(err != CL_SUCCESS) {
                std::cout << "enqueueNDRangeKernel failed: " << err << std::endl;
            }

            // read back buffer fill status
            err = queue.enqueueReadBuffer(stage.clFillCount, CL_TRUE, 0,
                                          stage.hostFillCount.size() * sizeof(cl_int), stage.hostFillCount.data(),
                                          nullptr, nullptr);
            if(err != CL_SUCCESS) {
                std::cout << "enqueueReadBuffer failed: " << err << std::endl;
            }

            for(uint32_t i = 0; i < N_STACKS; i++) {
                if(stage.hostFillCount.at(i) > BUF_THRESHOLD) {
                    stage_work_item item;
                    item.stageIdx = 1;
                    item.bufferIdx = i;
                    work_queue.push_front(item);
                }
            }

            // redo until buffer sufficiently filled
            continue;
        }

        // process remaining start conditions in OpenCL device buffers
        if(work_queue.empty()) {
#if 0
            // get some stats about stack fill levels, skip LAST stage, it has no stack
            for(size_t i = 0; i < (stages.size() - 1); i++) {
                auto& stage = stages.at(i);

                // read back buffer fill status
                err = queue.enqueueReadBuffer(stage.clFillCount, CL_TRUE, 0,
                                              stage.hostFillCount.size() * sizeof(cl_int), stage.hostFillCount.data(),
                                              nullptr, nullptr);
                if(err != CL_SUCCESS) {
                    std::cout << "enqueueReadBuffer failed: " << err << std::endl;
                }

                std::cout << "Stage " << std::to_string(stage.stageIdx) << " stack fill:" << std::left;
                for(size_t b = 0; b < N_STACKS; b++) {
                    std::cout << " " << std::setw(3) << stage.hostFillCount.at(b);
                }
                std::cout << std::endl;
            }
#endif
            // select first stage
            auto& stage = stages.at(0);

            if(stage.type == LAST1 || stage.type == LAST2) {
                std::cout << "Empty work queue on last stage" << std::endl;
                stages = std::vector<sieve_stage>(++stages.begin(), stages.end());
                done = true;
                continue;
            }
            // read back buffer fill status
            err = queue.enqueueReadBuffer(stage.clFillCount, CL_TRUE, 0,
                                          stage.hostFillCount.size() * sizeof(cl_int), stage.hostFillCount.data(),
                                          nullptr, nullptr);
            if(err != CL_SUCCESS) {
                std::cout << "enqueueReadBuffer failed: " << err << std::endl;
            }

            // remember if we queued tasks
            bool todo = false;
            for(uint32_t i = 0; i < N_STACKS; i++) {
                if(stage.hostFillCount.at(i) > 0) {
                    stage_work_item item;
                    item.stageIdx = 1;
                    item.bufferIdx = i;
                    work_queue.push_front(item);
                    todo = true;
                }
            }

            if(!todo) {
                // remove first stage since it's empty and unused now
                stages = std::vector<sieve_stage>(++stages.begin(), stages.end());
            }

            continue;
        }

        // get work item from queue
        auto item = work_queue.front();
        work_queue.pop_front();
        uint32_t curStageIdx  = item.stageIdx;
        auto& stage = stages.at(curStageIdx);

        // Only MID and LAST items can appear here
        // check if last stage
        if(curStageIdx == (stages.size() - 1)) {
            // select buffer
            stage.clKernel.setArg(1, item.bufferIdx);
            // Launch kernel on the compute device.
            err = queue.enqueueNDRangeKernel(stage.clKernel, cl::NullRange,
                                             cl::NDRange{N_STACKS}, cl::NDRange{std::min(WORKGROUP_SIZE, N_STACKS)},
                                             nullptr, &stage.clStageDone);

            if(err != CL_SUCCESS) {
                std::cout << "enqueueNDRangeKernel failed: " << err << std::endl;
            }

            queue.finish();

            // read back sum buffer
            err = queue.enqueueReadBuffer(stage.clSum, CL_TRUE, 0,
                                          hostOutputBuf.size() * sizeof(cl_uint), hostOutputBuf.data(),
                                          nullptr, nullptr);
            if(err != CL_SUCCESS) {
                std::cout << "enqueueReadBuffer failed: " << err << std::endl;
            }

            uint64_t intermediate_res = 0;
            for(uint32_t i = 0; i < N_STACKS; i++) {
                intermediate_res += hostOutputBuf[i];
            }

            result += intermediate_res;

        } else {
            // select buffer
            stage.clKernel.setArg(1, item.bufferIdx);
            // Launch kernel on the compute device.
            err = queue.enqueueNDRangeKernel(stage.clKernel, cl::NullRange,
                                             cl::NDRange{N_STACKS}, cl::NDRange{std::min(WORKGROUP_SIZE, N_STACKS)},
                                             nullptr, &stage.clStageDone);

            if(err != CL_SUCCESS) {
                std::cout << "enqueueNDRangeKernel failed: " << err << std::endl;
            }

            // read back buffer fill status
            err = queue.enqueueReadBuffer(stage.clFillCount, CL_TRUE, 0,
                                          stage.hostFillCount.size() * sizeof(cl_int), stage.hostFillCount.data(),
                                          nullptr, nullptr);
            if(err != CL_SUCCESS) {
                std::cout << "enqueueReadBuffer failed: " << err << std::endl;
            }

            for(uint32_t i = 0; i < N_STACKS; i++) {
                if(stage.hostFillCount.at(i) > BUF_THRESHOLD) {
                    stage_work_item item;
                    item.stageIdx = curStageIdx + 1;
                    item.bufferIdx = i;
                    work_queue.push_front(item);
                }
            }
        }
    }

    return result * 2;
}

