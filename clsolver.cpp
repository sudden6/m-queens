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

    err = platform.getDevices(CL_DEVICE_TYPE_GPU|
                              CL_DEVICE_TYPE_CPU, &devices);
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

    // Create command queue.
    queue = cl::CommandQueue(context, device, 0, &err);
    if(err != CL_SUCCESS) {
        std::cout << "failed to create command queue: " << err << std::endl;
    }

    // load source code
    std::ifstream sourcefile("clqueens_pre.cl");
    sourceStr = std::string((std::istreambuf_iterator<char>(sourcefile)),
                     std::istreambuf_iterator<char>());

}

constexpr uint_fast8_t MINN = 2;
constexpr uint_fast8_t MAXN = 29;

constexpr size_t N_STACKS = 64; // number of stacks
constexpr size_t WORKGROUP_SIZE = 64;   // number of threads that are run in parallel
constexpr size_t PLACED_PER_STAGE = 2;  // number of queens placed per sieve stage

/*
 * GPU_DEPTH defines how many rows should be left for the GPU to solve,
 * the previous ones have to be solved with the cpu.
 * With a too high GPU_DEPTH, solving a board takes too long and the
 * GPU is detected as "hung" by the driver and reset or the system crashes.
 */
constexpr uint_fast8_t GPU_DEPTH = 10;

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
                  << " -D IN_STACK_SIZE=" << std::to_string(stage.in_buf_size)
                  << " -D OUT_STACK_SIZE=" << std::to_string(stage.buf_size);

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

// +1 is the safety factor
void ClSolver::compute_stage_buf_size(sieve_stage& stage) {
    // TODO(sudden6): refine this
    stage.buf_size = stage.max_runs * stage.expansion;
}

void ClSolver::compute_buf_threshold(sieve_stage& stage) {
    // TODO(sudden6): refine this
    stage.buf_threshold = 0;
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
    compute_stage_buf_size(stage);
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
                                  N_STACKS * stage.buf_size * sizeof(start_condition), nullptr, &err);
    if(err != CL_SUCCESS) {
        std::cout << "cl::Buffer clStageBuf failed: " << err << std::endl;
        return false;
    }

    // Initialize stage buffer element count to zero
    stage.clFillCount = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                  N_STACKS * sizeof(cl_uint), nullptr, &err);
    if(err != CL_SUCCESS) {
        std::cout << "cl::Buffer clFillCount failed: " << err << std::endl;
        return false;
    }

    // zero fill count buffer
    err = queue.enqueueFillBuffer<cl_int>(stage.clFillCount, 0, 0, N_STACKS * sizeof(cl_uint));

    if(err != CL_SUCCESS) {
        std::cout << "enqueueFillBuffer failed: " << err << std::endl;
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
    stages.push_back(std::move(stage));
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
    stage.in_buf_size = prev_stage.buf_size;

    compute_expansion(stage);
    compute_stage_buf_size(stage);
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
                                  N_STACKS * stage.buf_size * sizeof(start_condition), nullptr, &err);
    if(err != CL_SUCCESS) {
        std::cout << "cl::Buffer clStageBuf failed: " << err << std::endl;
        return false;
    }

    // Initialize stage buffer element count to zero
    stage.clFillCount = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                  N_STACKS * sizeof(cl_uint), nullptr, &err);
    if(err != CL_SUCCESS) {
        std::cout << "cl::Buffer clFillCount failed: " << err << std::endl;
        return false;
    }

    // zero fill count buffer
    err = queue.enqueueFillBuffer<cl_uint>(stage.clFillCount, 0, 0, N_STACKS * sizeof(cl_uint));

    if(err != CL_SUCCESS) {
        std::cout << "enqueueFillBuffer failed: " << err << std::endl;
    }

    // create OpenCL kernels
    stage.clKernel = cl::Kernel(stage.program, "inter_step", &err);
    if(err != CL_SUCCESS) {
        std::cout << "cl::Kernel inter_step failed: " << err << std::endl;
    }
    // input buffer
    stage.clKernel.setArg(0, prev_stage.clStageBuf);
    // TODO(sudden6): refine this
    // max_runs
    stage.clKernel.setArg(1, 1);
    // output buffer
    stage.clKernel.setArg(2, stage.clStageBuf);
    // output buffer fill status
    stage.clKernel.setArg(3, stage.clFillCount);
    // input buffer fill status
    stage.clKernel.setArg(4, prev_stage.clFillCount);

    // insert into stages list
    stages.push_back(std::move(stage));
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
    stage.in_buf_size = prev_stage.buf_size;

    compute_expansion(stage);
    stage.buf_threshold = UINT32_MAX / stage.depth - 10;

    std::cout << "Creating final stage " << std::to_string(stage.index) << std::endl;
    /*
              << "  expansion: " << std::to_string(expansion) << std::endl
              << "  depth    : " << std::to_string(stage.depth) << std::endl
              << "  placed   : " << std::to_string(stage.placed) << std::endl;
    //*/


    if(!ClSolver::build_program(stage)) {
        return false;
    }

    // Initialize output sum buffer
    stage.clSum = cl::Buffer(context, CL_MEM_READ_WRITE,
                                  N_STACKS * sizeof(cl_uint), nullptr, &err);
    if(err != CL_SUCCESS) {
        std::cout << "cl::Buffer clSum failed: " << err << std::endl;
    }

    // zero fill sum buffer
    err = queue.enqueueFillBuffer<cl_uint>(stage.clSum, 0, 0, N_STACKS * sizeof(cl_uint));

    if(err != CL_SUCCESS) {
        std::cout << "enqueueFillBuffer failed: " << err << std::endl;
    }

    // create device kernel
    stage.clKernel = cl::Kernel(stage.program, "final_step", &err);
    if(err != CL_SUCCESS) {
        std::cout << "cl::Kernel final_step failed: " << err << std::endl;
    }
    // input buffer
    stage.clKernel.setArg(0, prev_stage.clStageBuf);
    // input buffer fill status
    stage.clKernel.setArg(1, prev_stage.clFillCount);
    // output sum buffer
    stage.clKernel.setArg(2, stage.clSum);

    // insert into stages list
    stages.push_back(std::move(stage));
    return true;
}

uint8_t ClSolver::queens_left() {
    const auto& last_stage = stages.back();
    return boardsize - last_stage.placed - last_stage.depth;
}

typedef cl_uint result_type;

constexpr size_t NUM_BATCHES = 4;
constexpr size_t NUM_CMDQUEUES = 2;

void ClSolver::fill_work_queue(sieve_stage& stage, int& next_stage) {

    if(stage.max_fill <= stage.buf_threshold) {
        return;
    }

    cl_int err = CL_SUCCESS;

    // map buffer to host for updating
    void * mapped_buffer = queue.enqueueMapBuffer(stage.clFillCount, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
                                 0, N_STACKS * sizeof(cl_uint),
                                 nullptr, nullptr, &err);
    if(err != CL_SUCCESS) {
        std::cout << "enqueueMapBuffer failed: " << err << std::endl;
    }

    cl_uint* fill = (cl_uint*) mapped_buffer;
    cl_uint max_fill = 0;
    cl_uint min_fill = UINT32_MAX;
    // TODO(sudden6): do this in an OpenCL kernel?
    for(uint32_t i = 0; i < N_STACKS; i++) {
        max_fill = std::max(fill[i], max_fill);
        min_fill = std::min(fill[i], min_fill);
    }

    if(max_fill > stage.buf_threshold) {
        next_stage++;
    }

    assert(max_fill <= stage.max_fill);

    stage.max_fill = max_fill;
    stage.min_fill = min_fill;

    // map buffer to device for working
    err = queue.enqueueUnmapMemObject(stage.clFillCount, mapped_buffer,
                                 nullptr, nullptr);
    if(err != CL_SUCCESS) {
        std::cout << "enqueueUnmapMemObject failed: " << err << std::endl;
    }

    //*
    // just ensure our buffers are not modified
    err = queue.enqueueBarrierWithWaitList();
    if(err != CL_SUCCESS) {
        std::cout << "enqueueBarrier failed: " << err << std::endl;
    }//*/
}

uint64_t ClSolver::solve_subboard(const std::vector<start_condition>::const_iterator &begin,
                                  const std::vector<start_condition>::const_iterator &end)
{
    cl_int err = CL_SUCCESS;

    if(begin == end) {
        return 0;
    }

    auto startIt = begin;

    // init presolver
    PreSolver pre(boardsize, placed, presolve_depth, *startIt);
    startIt++;

    uint32_t stage_mem = 0;
    for(size_t i = 0; i < (stages.size() - 1); i++) {
        stage_mem += stages.at(i).buf_size;
    }

    uint32_t mb_needed = (N_STACKS * stage_mem * 12)/(1024 * 1024);

    std::cout << "Memory needed: " << std::to_string(mb_needed) << std::endl;

    uint64_t result = 0;

    cl::Buffer clInputBuf;
    std::vector<start_condition> hostStartBuf{N_STACKS};
    std::vector<cl_uint> hostOutputBuf(N_STACKS, 0);     // store the result of the last stage
    int next_stage = 0;

    std::cout << "Entering crunch stage" << std::endl;

    // exit when all items in the buffers are removed
    while(true) {
        // fill step
        if(next_stage == 0 && !pre.empty()) {
            // insert new material at first sieve stage
            auto& stage = stages.at(0);
            // fill buffer
            auto hostBufIt = hostStartBuf.begin();
            while(hostBufIt != hostStartBuf.end()) {
                hostBufIt = pre.getNext(hostBufIt, hostStartBuf.cend());
                if(pre.empty() && (startIt == end)) {
                    // out of start conditions
                    std::cout << "Out of start conditions" << std::endl;
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
                continue;
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

            size_t local_size = WORKGROUP_SIZE;
            while(input_cnt % local_size != 0) {
                local_size /= 2;
            }

            // Launch kernel on the compute device.
            err = queue.enqueueNDRangeKernel(stage.clKernel, cl::NullRange,
                                             cl::NDRange{input_cnt},
                                             cl::NDRange{local_size},
                                             nullptr, &stage.clStageDone);

            if(err != CL_SUCCESS) {
                std::cout << "enqueueNDRangeKernel failed: " << err << std::endl;
            }

            stage.max_fill += stage.expansion;

            fill_work_queue(stage, next_stage);

            // redo until buffer sufficiently filled
            continue;
        }
        // process remaining start conditions in OpenCL device buffers
        if(next_stage == 0) {
            // select first stage
            auto& stage = stages.at(0);

            if(stage.type == STAGE_TYPE::LAST) {
                std::cout << "Empty work queue on last stage" << std::endl;
                // keep last stage, so we can read the sum buffer later
                break;
            }

            fill_work_queue(stage, next_stage);

            if(next_stage == 0) {
                if(stage.buf_threshold > 0) {
                    stage.buf_threshold /= 2;
                    // ensure multiple of workgroup size
                    stage.buf_threshold -= stage.buf_threshold % WORKGROUP_SIZE;
                    std::cout << "Reducing stage " << std::to_string(stage.index)
                              << " buf threshold to: " << std::to_string(stage.buf_threshold)
                              << std::endl;
                } else {
                    assert(stage.max_fill == 0);
                    std::cout << "Removing stage " << std::to_string(stage.index) << std::endl;
                    // remove first stage since it's empty and unused now
                    stages = std::move(std::vector<sieve_stage>(
                                           std::make_move_iterator(++stages.begin()),
                                           std::make_move_iterator(stages.end())));
                }
            }

            continue;
        }


        auto& stage = stages.at(next_stage);
        auto& prev_stage = stages.at(next_stage - 1);
        assert(stage.max_runs > 0);
        if(prev_stage.max_fill == 0) {
            next_stage--;
            continue;
        }

        if(stage.type == STAGE_TYPE::MID) {
            // set max_runs
            stage.clKernel.setArg(1, stage.max_runs);
        }


        // Launch kernel on the compute device.
        err = queue.enqueueNDRangeKernel(stage.clKernel, cl::NullRange,
                                         cl::NDRange{N_STACKS}, cl::NDRange{WORKGROUP_SIZE},
                                         nullptr, &stage.clStageDone);

        if(err != CL_SUCCESS) {
            std::cout << "enqueueNDRangeKernel failed: " << err << std::endl;
        }

        stage.max_fill += stage.expansion;
        prev_stage.max_fill -= stage.max_runs;

        // Only MID and LAST items can appear here
        // check if last stage
        if(stage.type == STAGE_TYPE::LAST) {
            if(stage.max_fill > stage.buf_threshold) {
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
                // reset max_fill counter
                stage.max_fill = 0;
            } else {
                // just ensure our buffers are not modified
                /*
                err = queue.enqueueBarrierWithWaitList();
                if(err != CL_SUCCESS) {
                    std::cout << "enqueueBarrier failed: " << err << std::endl;
                }//*/
            }
        } else {
            fill_work_queue(stage, next_stage);
        }
        // just ensure our buffers are not modified
        //*
        err = queue.enqueueBarrierWithWaitList();
        if(err != CL_SUCCESS) {
            std::cout << "enqueueBarrier failed: " << err << std::endl;
        }//*/
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

