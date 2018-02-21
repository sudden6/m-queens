#include "clsolver.h"
#include <ios>
#include <iostream>
#include <iterator>
#include <string>
#include <fstream>
#include <streambuf>
#include <sstream>
#include <queue>
#include "presolver.h"

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
constexpr uint_fast8_t GPU_DEPTH = 6;

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
                  << " -D PLACED=" <<std::to_string(gpu_depth);
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
constexpr size_t BATCH_SIZE = 2048;

typedef struct {
    cl::Kernel clKernel;
    cl::Buffer clStartBuf;
    cl::Buffer clOutputBuf;
    cl::Event clBatchDone;
    std::vector<start_condition> hostStartBuf;
    std::vector<cl_uint> hostOutputBuf;
    size_t size = 0;
} batch;

constexpr size_t NUM_BATCHES = 2;

uint64_t ClSolver::solve_subboard(start_condition &start)
{
    cl_int err = 0;

    // init presolver
    PreSolver pre(boardsize, placed, presolve_depth, start);

    // Create command queue.
    cl::CommandQueue cmdQueue(context, device, 0, &err);
    if(err != CL_SUCCESS) {
        std::cout << "CommandQueue failed: " << err << std::endl;
    }

    std::queue<batch> batQueue;
    uint64_t result = 0;

    while (!pre.empty()) {
        if(batQueue.size() >= NUM_BATCHES) {
            batch first = batQueue.front();
            first.clBatchDone.wait();
            for(size_t i = 0; i < first.size; i++) {
                result += first.hostOutputBuf[i];
            }
            batQueue.pop();
        }

        batch b;
        // create device kernel
        b.clKernel = cl::Kernel(program, "solve_subboard", &err);
        auto& kernel = b.clKernel;
        if(err != CL_SUCCESS) {
            std::cout << "cl::Kernel failed: " << err << std::endl;
        }

        b.hostStartBuf = pre.getNext(BATCH_SIZE);
        auto& startBuf = b.hostStartBuf;
        b.size = startBuf.size();
        const auto& batchSize = b.size;

        // Allocate device buffers and transfer input data to device.
        b.clStartBuf = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            batchSize * sizeof(start_condition), startBuf.data(), &err);
        if(err != CL_SUCCESS) {
            std::cout << "cl::Buffer start_buf failed: " << err << std::endl;
        }

        auto& clStartBuf = b.clStartBuf;

        // result buffer
        b.hostOutputBuf = std::vector<cl_uint>(batchSize, 0);
        auto& outBuf = b.hostOutputBuf;
        b.clOutputBuf = cl::Buffer(context, CL_MEM_WRITE_ONLY,
            batchSize * sizeof(cl_uint), nullptr, &err);
        if(err != CL_SUCCESS) {
            std::cout << "cl::Buffer results_buf failed: " << err << std::endl;
        }

        auto& clOutputBuf = b.clOutputBuf;

        // Set kernel parameters.
        err = kernel.setArg(0, clStartBuf);
        if(err != CL_SUCCESS) {
            std::cout << "solve_subboard.setArg(0 failed: " << err << std::endl;
        }

        err = kernel.setArg(1, clOutputBuf);
        if(err != CL_SUCCESS) {
            std::cout << "solve_subboard.setArg(1 failed: " << err << std::endl;
        }

        // Launch kernel on the compute device.
        err = cmdQueue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange{batchSize},
                                         cl::NullRange);
        if(err != CL_SUCCESS) {
            std::cout << "enqueueNDRangeKernel failed: " << err << std::endl;
        }

        // Get result back to host.
        err = cmdQueue.enqueueReadBuffer(clOutputBuf, CL_FALSE, 0,
                                      batchSize * sizeof(cl_uint), outBuf.data(), nullptr, &b.clBatchDone);
        if(err != CL_SUCCESS) {
            std::cout << "enqueueReadBuffer failed: " << err << std::endl;
        }

        // send to device
        cmdQueue.flush();
        batQueue.push(b);
    }

    cmdQueue.finish();

    while (!batQueue.empty()) {
        batch first = batQueue.front();
        first.clBatchDone.wait();
        for(size_t i = 0; i < first.size; i++) {
            result += first.hostOutputBuf[i];
        }
        batQueue.pop();
    }

    return result * 2;
}



