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

// should be a multiple of 64 at least for AMD GPUs
// ideally would be CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE
// bigger BATCH_SIZE means higher memory usage
constexpr size_t BATCH_SIZE = 1 << 16;

typedef struct {
    cl::Kernel clKernel;
    cl::Buffer clStartBuf;
    cl::Buffer clOutputBuf;
    cl::Event clBatchDone;
    std::vector<cl::Event> clStartKernel;
    std::vector<cl::Event> clReadResult;
    std::vector<start_condition> hostStartBuf;
    std::vector<cl_uint> hostOutputBuf;
    size_t size = 0;
} batch;

constexpr size_t NUM_BATCHES = 16;

uint64_t ClSolver::solve_subboard(start_condition &start)
{
    cl_int err = CL_SUCCESS;

    // init presolver
    PreSolver pre(boardsize, placed, presolve_depth, start);

    // Create command queue.
    cl::CommandQueue cmdQueue(context, device, 0, &err);
    if(err != CL_SUCCESS) {
        std::cout << "CommandQueue failed: " << err << std::endl;
    }

    // buffer
    batch batches[NUM_BATCHES];
    std::list<size_t> running;
    std::list<size_t> complete;

    for(int i = 0; i < NUM_BATCHES; i++) {
        batch& b = batches[i];
        // add this slot to the free queue
        complete.push_back(i);
        // create device kernel
        b.clKernel = cl::Kernel(program, "solve_subboard", &err);
        if(err != CL_SUCCESS) {
            std::cout << "cl::Kernel failed: " << err << std::endl;
        }
        // Allocate start condition buffer on device
        b.clStartBuf = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
            BATCH_SIZE * sizeof(start_condition), nullptr, &err);
        if(err != CL_SUCCESS) {
            std::cout << "cl::Buffer start_buf failed: " << err << std::endl;
        }

        // Allocate result buffer on device
        b.clOutputBuf = cl::Buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
            BATCH_SIZE * sizeof(cl_uint), nullptr, &err);
        if(err != CL_SUCCESS) {
            std::cout << "cl::Buffer results_buf failed: " << err << std::endl;
        }

        // Allocate result buffer on host
        b.hostOutputBuf = std::vector<cl_uint>(BATCH_SIZE, 0);

        // Set kernel parameters.
        err = b.clKernel.setArg(0, b.clStartBuf);
        if(err != CL_SUCCESS) {
            std::cout << "solve_subboard.setArg(0 failed: " << err << std::endl;
        }

        err = b.clKernel.setArg(1, b.clOutputBuf);
        if(err != CL_SUCCESS) {
            std::cout << "solve_subboard.setArg(1 failed: " << err << std::endl;
        }
    }

    uint64_t result = 0;

    while (!pre.empty()) {
        // move idx from free queue to busy queue
        size_t cur = complete.front();
        complete.pop_front();
        running.push_back(cur);
        auto& b = batches[cur];

        b.hostStartBuf = pre.getNext(BATCH_SIZE);
        b.size = b.hostStartBuf.size();
        const auto& batchSize = b.size;

        // Transfer start buffer to device
        err = cmdQueue.enqueueWriteBuffer(b.clStartBuf, CL_FALSE, 0,
                                          batchSize * sizeof(start_condition), b.hostStartBuf.data(),
                                          nullptr, &b.clStartKernel[0]);

        if(err != CL_SUCCESS) {
            std::cout << "Failed to transfer start buffer: " << err << std::endl;
        }

        // Launch kernel on the compute device.
        err = cmdQueue.enqueueNDRangeKernel(b.clKernel, cl::NullRange,
                                            cl::NDRange{batchSize}, cl::NullRange,
                                            &b.clStartKernel, &b.clReadResult[0]);
        if(err != CL_SUCCESS) {
            std::cout << "enqueueNDRangeKernel failed: " << err << std::endl;
        }

        // Get result back to host.
        err = cmdQueue.enqueueReadBuffer(b.clOutputBuf, CL_FALSE, 0,
                                         batchSize * sizeof(cl_uint), b.hostOutputBuf.data(),
                                         &b.clReadResult, &b.clBatchDone);
        if(err != CL_SUCCESS) {
            std::cout << "enqueueReadBuffer failed: " << err << std::endl;
        }

        // poll events to find free slot

        // wait for free slot
        auto it = running.begin();
        while (complete.empty()) {
            // loop endless over running batches
            if(it == running.end()) {
                it = running.begin();
                // TODO(sudden6): sleep?
            }

            size_t idx = *it;
            batch& c = batches[idx];
            // check if batch finished
            if(c.clBatchDone.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>() == CL_COMPLETE) {
                // get data from completed batch
                for(size_t i = 0; i < c.size; i++) {
                    result += c.hostOutputBuf[i];
                }
                // remove from running batches
                running.erase(it);
                // add to completed batches
                complete.push_back(idx);
            }
        }
    }

    auto it = running.begin();
    while (!running.empty()) {
        // loop endless over running batches
        if(it == running.end()) {
            it = running.begin();
            // TODO(sudden6): sleep?
        }

        size_t idx = *it;
        batch& c = batches[idx];
        // check if batch finished
        if(c.clBatchDone.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>() == CL_COMPLETE) {
            // get data from completed batch
            for(size_t i = 0; i < c.size; i++) {
                result += c.hostOutputBuf[i];
            }
            // remove from running batches
            it = running.erase(it);
        }
    }

    return result * 2;
}



