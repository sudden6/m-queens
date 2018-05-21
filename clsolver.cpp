#include "clsolver.h"
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

    err = platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
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
    std::ifstream sourcefile("clqueens_amd.cl");
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
/*
 * GPU_DEPTH defines how many rows should be left for the GPU to solve,
 * the previous ones have to be solved with the cpu.
 * With a too high GPU_DEPTH, solving a board takes too long and the
 * GPU is detected as "hung" by the driver and reset or the system crashes.
 */
constexpr uint_fast8_t GPU_DEPTH = 9;

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
                  << " -D PLACED=" <<std::to_string(boardsize - gpu_depth);
    std::string options = optionsStream.str();

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
    std::vector<cl_ushort> hostFillCount;
    STAGE_TYPE type;
} sieve_stage;

typedef struct {
    uint32_t bufferIdx;     // must be 32 Bit, to match OpenCL kernel
    uint32_t stageIdx;
} stage_work_item;

constexpr size_t PLACED_PER_STAGE = 2;  // number of queens placed per sieve stage
constexpr size_t WORKGROUP_SIZE = 64;   // number of threads that are run in parallel
constexpr size_t GLOBAL_WORKSIZE = 256; // number of elements per buffer
constexpr size_t STAGE_SIZE = WORKGROUP_SIZE * GLOBAL_WORKSIZE;

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
    // TODO: don't hardcode
    uint8_t queens_left = GPU_DEPTH; // number of queens to place till board is full
    // we need at least 2 (first stage) + 2 (mid stage) + 1 (last1 stage) queens left
    if(queens_left < 5) {
        std::cout << "not enough queens left" << std::endl;
        return 0;
    }

    // create first sieve stage
    sieve_stage first{};
    first.type = FIRST;
    stages.push_back(first);
    queens_left -= 2;

    // create middle sieve stages
    while(queens_left > 2) {
        sieve_stage mid{};
        mid.type = MID;
        stages.push_back(mid);
        queens_left -= 2;
    }

    // create last sieve stage
    sieve_stage last{};
    if(queens_left == 1) {
        last.type = LAST1;
    } else if (queens_left == 2) {
        last.type = LAST2;
    } else {
        std::cout << "wrong number of queens left" << std::endl;
        return 0;
    }

    stages.push_back(last);

    std::cout << "Number of stages: " << stages.size();

    // initialize OpenCL stuff
    for(size_t i = 0; i < stages.size(); i++) {
        sieve_stage& stage = stages.at(i);
        // Initialize stage buffer
        if(stage.type != LAST1 || stage.type != LAST2) {
            stage.clStageBuf = cl::Buffer(context, CL_MEM_READ_WRITE,
                                          STAGE_SIZE * sizeof(start_condition), nullptr, &err);
            if(err != CL_SUCCESS) {
                std::cout << "cl::Buffer clStageBuf failed: " << err << std::endl;
            }

            // host fill count buffer
            stage.hostFillCount = std::vector<cl_ushort>(WORKGROUP_SIZE, 0);

            // Initialize stage buffer element count to zero
            stage.clFillCount = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                          WORKGROUP_SIZE * sizeof(cl_ushort), stage.hostFillCount.data(), &err);
            if(err != CL_SUCCESS) {
                std::cout << "cl::Buffer clFillCount failed: " << err << std::endl;
            }
        }

        // Initialize output buffer
        if(stage.type == LAST1 || stage.type == LAST2) {
            stage.clSum = cl::Buffer(context, CL_MEM_READ_WRITE,
                                          WORKGROUP_SIZE * sizeof(result_type), nullptr, &err);
            if(err != CL_SUCCESS) {
                std::cout << "cl::Buffer clSum failed: " << err << std::endl;
            }

        }

        const auto prev_stage_idx = i - 1;
        // TODO(sudden6): call actual kernels
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
    constexpr size_t BUF_THRESHOLD = 128;

    uint64_t result = 0;

    std::list<stage_work_item> work_queue{};
    cl::Buffer clInputBuf;
    std::vector<start_condition> hostStartBuf{WORKGROUP_SIZE};
    std::vector<result_type> hostOutputBuf{WORKGROUP_SIZE, 0};     // store the result of the last stage

    while (!pre.empty()) {

        if(work_queue.empty()) {
            // insert new material at first sieve stage
            auto& stage = stages.at(0);
            // fill buffer
            auto hostBufIt = hostStartBuf.begin();
            while(hostBufIt != hostStartBuf.end()) {
                hostBufIt = pre.getNext(hostBufIt, hostStartBuf.cend());
                if(pre.empty() && (startIt == start.end())) {
                    break;
                }
                if(pre.empty()) {
                    pre = PreSolver(boardsize, placed, presolve_depth, *startIt);
                    startIt++;
                }
            }

            // TODO(sudden6): handle case where we run out of start_conditions before
            //                filling the buffer
            //                use filler values? or cpu solver?

            // upload input data to device
            clInputBuf = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                    WORKGROUP_SIZE * sizeof(start_condition), hostStartBuf.data(), &err);
            if(err != CL_SUCCESS) {
                std::cout << "cl::Buffer clInputBuf failed: " << err << std::endl;
            }

            // input buffer
            stage.clKernel.setArg(0, clInputBuf);

            // Launch kernel on the compute device.
            err = queue.enqueueNDRangeKernel(stage.clKernel, cl::NullRange,
                                             cl::NDRange{WORKGROUP_SIZE}, cl::NullRange,
                                             nullptr, &stage.clStageDone);

            if(err != CL_SUCCESS) {
                std::cout << "enqueueNDRangeKernel failed: " << err << std::endl;
            }

            // read back buffer fill status
            err = queue.enqueueReadBuffer(stage.clFillCount, CL_TRUE, 0,
                                          WORKGROUP_SIZE * sizeof(cl_ushort), stage.hostFillCount.data(),
                                          nullptr, nullptr);
            if(err != CL_SUCCESS) {
                std::cout << "enqueueReadBuffer failed: " << err << std::endl;
            }

            for(uint32_t i = 0; i < WORKGROUP_SIZE; i++) {
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

        // get work item from queue
        auto item = work_queue.front();
        work_queue.pop_front();
        uint32_t curStageIdx  = item.stageIdx;
        auto& stage = stages.at(curStageIdx);

        // Only mid and last items can appear here
        // check if last stage
        if(curStageIdx == (stages.size() - 1)) {
            // select buffer
            stage.clKernel.setArg(1, &item.bufferIdx);
            // Launch kernel on the compute device.
            err = queue.enqueueNDRangeKernel(stage.clKernel, cl::NullRange,
                                             cl::NDRange{WORKGROUP_SIZE}, cl::NullRange,
                                             nullptr, &stage.clStageDone);

            if(err != CL_SUCCESS) {
                std::cout << "enqueueNDRangeKernel failed: " << err << std::endl;
            }

            // read back sum buffer
            err = queue.enqueueReadBuffer(stage.clSum, CL_TRUE, 0,
                                          WORKGROUP_SIZE * sizeof(result_type), hostOutputBuf.data(),
                                          nullptr, nullptr);
            if(err != CL_SUCCESS) {
                std::cout << "enqueueReadBuffer failed: " << err << std::endl;
            }

            uint64_t intermediate_res = 0;
            for(uint32_t i = 0; i < WORKGROUP_SIZE; i++) {
                intermediate_res += hostOutputBuf[i];
            }

        } else {
            // select buffer
            stage.clKernel.setArg(1, &item.bufferIdx);
            // Launch kernel on the compute device.
            err = queue.enqueueNDRangeKernel(stage.clKernel, cl::NullRange,
                                             cl::NDRange{WORKGROUP_SIZE}, cl::NullRange,
                                             nullptr, &stage.clStageDone);

            if(err != CL_SUCCESS) {
                std::cout << "enqueueNDRangeKernel failed: " << err << std::endl;
            }

            // read back buffer fill status
            err = queue.enqueueReadBuffer(stage.clFillCount, CL_TRUE, 0,
                                          WORKGROUP_SIZE * sizeof(cl_ushort), stage.hostFillCount.data(),
                                          nullptr, nullptr);
            if(err != CL_SUCCESS) {
                std::cout << "enqueueReadBuffer failed: " << err << std::endl;
            }

            for(uint32_t i = 0; i < WORKGROUP_SIZE; i++) {
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

