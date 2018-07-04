#include "clsolver.h"
#include <cassert>
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
    sourceStr = std::string((std::istreambuf_iterator<char>(sourcefile)),
                     std::istreambuf_iterator<char>());

}

constexpr uint_fast8_t MINN = 2;
constexpr uint_fast8_t MAXN = 29;

constexpr size_t N_STACKS = 9216; // number of stacks
constexpr size_t WORKGROUP_SIZE = 64;   // number of threads that are run in parallel
constexpr size_t STACK_SIZE = N_STACKS+320;      // number of elements in a stack
constexpr size_t PLACED_PER_STAGE = 2;  // number of queens placed per sieve stage

/*
 * GPU_DEPTH defines how many rows should be left for the GPU to solve,
 * the previous ones have to be solved with the cpu.
 * With a too high GPU_DEPTH, solving a board takes too long and the
 * GPU is detected as "hung" by the driver and reset or the system crashes.
 */
constexpr uint_fast8_t GPU_DEPTH = 8;

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

    // create first sieve stage
    if(!first_stage()) {
        return 0;
    }

    // create following sieve stages
    while(queens_left() > PLACED_PER_STAGE) {
        append_mid_stage();
    }

    // create last sieve stage
    append_last_stage();

    uint32_t max_expansion = 0;
    // compute max expansion factors
    for(auto& stage : stages) {
        max_expansion = std::max(stage.expansion, max_expansion);
    }

    std::cout << "Number of stages: " << stages.size() << std::endl;
    std::cout << "Maximum expansion: " << max_expansion << std::endl;

    return true;
}

bool ClSolver::build_program(sieve_stage& stage) {

    cl_int err = CL_SUCCESS;
    std::ostringstream optionsStream;
    optionsStream << "-D N=" << std::to_string(boardsize)
                  << " -D PLACED=" << std::to_string(stage.placed)
                  << " -D N_STACKS=" << std::to_string(N_STACKS)
                  << " -D WORKGROUP_SIZE=" << std::to_string(WORKGROUP_SIZE)
                  << " -D DEPTH=" << std::to_string(stage.depth)
                  << " -D STAGE_IDX=" << std::to_string(stage.index)
                  << " -D EXPANSION=" << std::to_string(stage.expansion)
                  << " -D STACK_SIZE=" << std::to_string(STACK_SIZE);

    std::string options = optionsStream.str();

    std::cout << "OpenCL Kernel Options: " << options << std::endl;

    // create OpenCL program
    stage.program = cl::Program(context, sourceStr, false, &err);
    if(err != CL_SUCCESS) {
        std::cout << "cl::Program failed" << std::endl;
        return false;
    }

    cl_int builderr = stage.program.build(options.c_str());
    if(builderr != CL_SUCCESS) {
        std::cout << "program.build failed: " << builderr << std::endl;
        cl_int err = 0;
        std::cout << "OpenCL build log:" << std::endl;
        auto buildlog = stage.program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device, &err);
        std::cout << buildlog << std::endl;
        if(err != CL_SUCCESS) {
            std::cout << "getBuildInfo<CL_PROGRAM_BUILD_LOG> failed" << std::endl;
        }
        return false;
    }
    return true;
}

void ClSolver::compute_expansion(sieve_stage& stage) {
    // compute expansion factor
    uint32_t expansion = 1;
    for(uint32_t i = 0; i < stage.depth; i++) {
        expansion *= boardsize - stage.placed - i;
    }
    stage.expansion = expansion;
}

void ClSolver::compute_buf_threshold(sieve_stage& stage) {
    stage.buf_threshold = STACK_SIZE - stage.expansion;
}

bool ClSolver::first_stage() {

    cl_int err = CL_SUCCESS;
    sieve_stage stage;

    stage.type = STAGE_TYPE::FIRST;
    stage.placed = placed + presolve_depth;
    stage.index = 0;
    stage.depth = PLACED_PER_STAGE;

    assert(stage.placed < boardsize);

    compute_expansion(stage);
    compute_buf_threshold(stage);

    std::cout << "Creating first stage" << std::endl;
    /*
              << "  expansion: " << std::to_string(expansion) << std::endl
              << "  depth    : " << std::to_string(stage.depth) << std::endl
              << "  placed   : " << std::to_string(stage.placed) << std::endl;
    //*/


    if(!ClSolver::build_program(stage)) {
        return false;
    }

    // create stage buffer
    stage.clStageBuf = cl::Buffer(context, CL_MEM_READ_WRITE,
                                  N_STACKS * STACK_SIZE * sizeof(start_condition), nullptr, &err);
    if(err != CL_SUCCESS) {
        std::cout << "cl::Buffer clStageBuf failed: " << err << std::endl;
        return false;
    }

    // host fill count buffer
    stage.hostFillCount = std::vector<cl_int>(N_STACKS, 0);

    // Initialize stage buffer element count to zero
    stage.clFillCount = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                  N_STACKS * sizeof(cl_int), stage.hostFillCount.data(), &err);
    if(err != CL_SUCCESS) {
        std::cout << "cl::Buffer clFillCount failed: " << err << std::endl;
        return false;
    }

    // create device kernel
    stage.clKernel = cl::Kernel(stage.program, "first_step", &err);
    if(err != CL_SUCCESS) {
        std::cout << "cl::Kernel first_step failed: " << err << std::endl;
    }
    // input buffer
    // arg 0 set later when input data known
    // output buffer
    stage.clKernel.setArg(1, stage.clStageBuf);
    // output buffer fill status
    stage.clKernel.setArg(2, stage.clFillCount);

    // insert into stages list
    stages.push_back(stage);
    return true;
}

bool ClSolver::append_mid_stage() {
    cl_int err = CL_SUCCESS;
    assert(stages.size() > 0);

    const sieve_stage& prev_stage = stages.back();

    sieve_stage stage;

    stage.type = STAGE_TYPE::MID;
    stage.placed = prev_stage.placed + prev_stage.depth;
    stage.index = stages.size();
    stage.depth = PLACED_PER_STAGE;

    compute_expansion(stage);
    compute_buf_threshold(stage);

    std::cout << "Creating mid stage " << std::to_string(stage.index) << std::endl;
    /*
              << "  expansion: " << std::to_string(expansion) << std::endl
              << "  depth    : " << std::to_string(stage.depth) << std::endl
              << "  placed   : " << std::to_string(stage.placed) << std::endl;
    //*/


    if(!ClSolver::build_program(stage)) {
        return false;
    }

    // create stage buffer
    stage.clStageBuf = cl::Buffer(context, CL_MEM_READ_WRITE,
                                  N_STACKS * STACK_SIZE * sizeof(start_condition), nullptr, &err);
    if(err != CL_SUCCESS) {
        std::cout << "cl::Buffer clStageBuf failed: " << err << std::endl;
        return false;
    }

    // host fill count buffer
    stage.hostFillCount = std::vector<cl_int>(N_STACKS, 0);

    // Initialize stage buffer element count to zero
    stage.clFillCount = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                  N_STACKS * sizeof(cl_int), stage.hostFillCount.data(), &err);
    if(err != CL_SUCCESS) {
        std::cout << "cl::Buffer clFillCount failed: " << err << std::endl;
        return false;
    }

    // create OpenCL kernels
    stage.clKernel = cl::Kernel(stage.program, "inter_step", &err);
    if(err != CL_SUCCESS) {
        std::cout << "cl::Kernel inter_step failed: " << err << std::endl;
    }
    // input buffer
    stage.clKernel.setArg(0, prev_stage.clStageBuf);
    // offset
    // arg 1 set later
    // output buffer
    stage.clKernel.setArg(2, stage.clStageBuf);
    // output buffer fill status
    stage.clKernel.setArg(3, stage.clFillCount);
    // input buffer fill status
    stage.clKernel.setArg(4, prev_stage.clFillCount);

    // insert into stages list
    stages.push_back(stage);
    return true;
}

bool ClSolver::append_last_stage() {
    cl_int err = CL_SUCCESS;
    assert(stages.size() > 0);

    const sieve_stage& prev_stage = stages.back();
    const int queens_left = boardsize - (prev_stage.placed + prev_stage.depth);

    assert(queens_left > 0);
    assert(queens_left < 3);

    sieve_stage stage;

    stage.type = STAGE_TYPE::LAST;
    stage.placed = prev_stage.placed + prev_stage.depth;
    stage.index = stages.size();
    stage.depth = queens_left;

    compute_expansion(stage);
    compute_buf_threshold(stage);

    std::cout << "Creating final stage " << std::to_string(stage.index) << std::endl;
    /*
              << "  expansion: " << std::to_string(expansion) << std::endl
              << "  depth    : " << std::to_string(stage.depth) << std::endl
              << "  placed   : " << std::to_string(stage.placed) << std::endl;
    //*/


    if(!ClSolver::build_program(stage)) {
        return false;
    }

    std::vector<cl_uint> zeroBuffer(N_STACKS, 0);

    // Initialize output sum buffer
    stage.clSum = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                  N_STACKS * sizeof(cl_uint), zeroBuffer.data(), &err);
    if(err != CL_SUCCESS) {
        std::cout << "cl::Buffer clSum failed: " << err << std::endl;
    }

    // create device kernel
    stage.clKernel = cl::Kernel(stage.program, "final_step", &err);
    if(err != CL_SUCCESS) {
        std::cout << "cl::Kernel final_step failed: " << err << std::endl;
    }
    // input buffer
    stage.clKernel.setArg(0, prev_stage.clStageBuf);
    // offset
    // arg 1 set later
    // input buffer fill status
    stage.clKernel.setArg(2, prev_stage.clFillCount);
    // output sum buffer
    stage.clKernel.setArg(3, stage.clSum);

    // insert into stages list
    stages.push_back(stage);
    return true;
}

uint8_t ClSolver::queens_left() {
    const auto& last_stage = stages.back();
    return boardsize - last_stage.placed - last_stage.depth;
}

typedef cl_uint result_type;

constexpr size_t NUM_BATCHES = 4;
constexpr size_t NUM_CMDQUEUES = 2;

typedef struct {
    cl_uint bufferIdx;     // must be 32 Bit, to match OpenCL kernel
    uint32_t stageIdx;
} stage_work_item;

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

    uint64_t result = 0;

    std::list<stage_work_item> work_queue{};
    cl::Buffer clInputBuf;
    std::vector<cl_uint> zeroBuffer(N_STACKS, 0);        // zero buffer to
    std::vector<start_condition> hostStartBuf{N_STACKS};
    std::vector<cl_uint> hostOutputBuf(N_STACKS, 0);     // store the result of the last stage

    std::cout << "Entering crunch stage" << std::endl;
    // exit when all start conditions from the pre solver are handled
    while (true) {
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

            if(input_cnt > N_STACKS) {
                std::cout << "Too much input" << std::endl;
            }

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
                if(stage.hostFillCount.at(i) > stage.buf_threshold) {
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
            break;
        }

        // get work item from queue
        auto item = work_queue.front();
        work_queue.pop_front();
        uint32_t curStageIdx  = item.stageIdx;
        auto& stage = stages.at(curStageIdx);

        // select buffer
        stage.clKernel.setArg(1, item.bufferIdx);
        // Launch kernel on the compute device.
        err = queue.enqueueNDRangeKernel(stage.clKernel, cl::NullRange,
                                         cl::NDRange{N_STACKS}, cl::NDRange{std::min(WORKGROUP_SIZE, N_STACKS)},
                                         nullptr, &stage.clStageDone);

        if(err != CL_SUCCESS) {
            std::cout << "enqueueNDRangeKernel failed: " << err << std::endl;
        }

        // Only MID and LAST items can appear here
        // check if last stage
        if(stage.type == STAGE_TYPE::LAST) {
            const uint32_t runs_threshold = UINT32_MAX / stage.depth - 10;
            if(stage.runs > runs_threshold) {
                // read back sum buffer
                std::cout << "reading sum buffer" << std::endl;
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

                // zero sum buffer
                err = queue.enqueueFillBuffer<cl_uint>(stage.clSum, 0, 0, hostOutputBuf.size() * sizeof(cl_uint));

                if(err != CL_SUCCESS) {
                    std::cout << "enqueueFillBuffer failed: " << err << std::endl;
                }

                // reset run counter
                stage.runs = 0;
            } else {
                stage.runs += stage.expansion;
                // just ensure our buffers are not modified
                err = queue.enqueueBarrierWithWaitList();
                if(err != CL_SUCCESS) {
                    std::cout << "enqueueBarrier failed: " << err << std::endl;
                }
            }

        } else {
            // read back buffer fill status
            err = queue.enqueueReadBuffer(stage.clFillCount, CL_TRUE, 0,
                                          stage.hostFillCount.size() * sizeof(cl_int), stage.hostFillCount.data(),
                                          nullptr, nullptr);
            if(err != CL_SUCCESS) {
                std::cout << "enqueueReadBuffer failed: " << err << std::endl;
            }

            for(uint32_t i = 0; i < N_STACKS; i++) {
                if(stage.hostFillCount.at(i) > stage.buf_threshold) {
                    stage_work_item item;
                    item.stageIdx = curStageIdx + 1;
                    item.bufferIdx = i;
                    work_queue.push_front(item);
                }
            }
        }
    }

    // ensure all buffer access is finished
    queue.enqueueBarrierWithWaitList();

    std::cout << "Entering cleanup stage" << std::endl;
    // exit when all items in the buffers are removed
    while(true) {
        // process remaining start conditions in OpenCL device buffers
        if(work_queue.empty()) {
            // select first stage
            auto& stage = stages.at(0);

            if(stage.type == STAGE_TYPE::LAST) {
                std::cout << "Empty work queue on last stage" << std::endl;
                // keep last stage, so we can read the sum buffer later
                break;
            }
            // read back buffer fill status
            err = queue.enqueueReadBuffer(stage.clFillCount, CL_TRUE, 0,
                                          stage.hostFillCount.size() * sizeof(cl_int), stage.hostFillCount.data(),
                                          nullptr, nullptr);
            if(err != CL_SUCCESS) {
                std::cout << "enqueueReadBuffer failed: " << err << std::endl;
            }

            for(uint32_t i = 0; i < N_STACKS; i++) {
                if(stage.hostFillCount.at(i) > 0) {
                    stage_work_item item;
                    item.stageIdx = 1;
                    item.bufferIdx = i;
                    work_queue.push_front(item);
                }
            }

            if(work_queue.empty()) {
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

        // select buffer
        stage.clKernel.setArg(1, item.bufferIdx);
        // Launch kernel on the compute device.
        err = queue.enqueueNDRangeKernel(stage.clKernel, cl::NullRange,
                                         cl::NDRange{N_STACKS}, cl::NDRange{std::min(WORKGROUP_SIZE, N_STACKS)},
                                         nullptr, &stage.clStageDone);

        if(err != CL_SUCCESS) {
            std::cout << "enqueueNDRangeKernel failed: " << err << std::endl;
        }

        // Only MID and LAST items can appear here
        // check if last stage
        if(stage.type == STAGE_TYPE::LAST) {
            const uint32_t runs_threshold = UINT32_MAX / stage.depth - 10;
            if(stage.runs > runs_threshold) {
                // read back sum buffer
                std::cout << "reading sum buffer" << std::endl;
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

                // zero sum buffer
                err = queue.enqueueFillBuffer<cl_uint>(stage.clSum, 0, 0, hostOutputBuf.size() * sizeof(cl_uint));

                if(err != CL_SUCCESS) {
                    std::cout << "enqueueFillBuffer failed: " << err << std::endl;
                }
                // reset run counter
                stage.runs = 0;
            } else {
                stage.runs += stage.expansion;
                // just ensure our buffers are not modified
                err = queue.enqueueBarrierWithWaitList();
                if(err != CL_SUCCESS) {
                    std::cout << "enqueueBarrier failed: " << err << std::endl;
                }
            }
        } else {
            // read back buffer fill status
            err = queue.enqueueReadBuffer(stage.clFillCount, CL_TRUE, 0,
                                          stage.hostFillCount.size() * sizeof(cl_int), stage.hostFillCount.data(),
                                          nullptr, nullptr);
            if(err != CL_SUCCESS) {
                std::cout << "enqueueReadBuffer failed: " << err << std::endl;
            }

            for(uint32_t i = 0; i < N_STACKS; i++) {
                if(stage.hostFillCount.at(i) > stage.buf_threshold) {
                    stage_work_item item;
                    item.stageIdx = curStageIdx + 1;
                    item.bufferIdx = i;
                    work_queue.push_front(item);
                }
            }
        }
    }

    std::cout << "reading sum buffer" << std::endl;
    err = queue.enqueueReadBuffer(stages.back().clSum, CL_TRUE, 0,
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

    return result * 2;
}

