#include <cassert>
#include "clsolver.h"
#include <ctime>
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
#include <CL/cl.h>
#include <CL/cl.hpp>

ClSolver::ClSolver()
{
}

constexpr uint_fast8_t MINN = 2;
constexpr uint_fast8_t MAXN = 29;
/*
 * GPU_DEPTH defines how many rows should be left for the GPU to solve,
 * the previous ones have to be solved with the cpu.
 * With a too high GPU_DEPTH, solving a board takes too long and the
 * GPU is detected as "hung" by the driver and reset or the system crashes.
 */
constexpr uint_fast8_t GPU_DEPTH = 7;
constexpr size_t WORKGROUP_SIZE = 64;

uint64_t ClSolver::expansion(uint8_t boardsize, uint8_t cur_idx, uint8_t depth) {
    assert(boardsize > cur_idx);
    assert(boardsize >= (cur_idx + depth));
    uint64_t start = 1;
    for(uint8_t i = 0; i < depth; i++) {
        start *= boardsize - (cur_idx + i);
    }
    return start;
}

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
    this->gpu_presolve_depth = 2;
    if((boardsize - placed) < GPU_DEPTH) {
        gpu_depth = boardsize - placed;
        presolve_depth = 0;
    } else {
        presolve_depth = boardsize - placed - GPU_DEPTH;
    }

    // Need at least 1 step presolve and 1 step solve
    if(gpu_depth < 2) {
        return false;
    }

    if(presolve_depth > 0) {
        this->gpu_presolve_depth = std::min(presolve_depth, gpu_presolve_depth);
        presolve_depth -= gpu_presolve_depth;
    } else {
        if(gpu_depth < 3) {
            this->gpu_presolve_depth = 1;
        }
        gpu_depth -= this->gpu_presolve_depth;
    }

    this->gpu_presolve_expansion = expansion(boardsize, placed + presolve_depth, gpu_presolve_depth);

    std::ostringstream optionsStream;
    optionsStream << "-D N=" << std::to_string(boardsize)
                  << " -D PLACED=" <<std::to_string(boardsize - gpu_depth)
                  << " -D DEPTH=" <<std::to_string(gpu_depth)
                  << " -D WG_SIZE=" <<std::to_string(WORKGROUP_SIZE)
                  << " -D PRE_END=" <<std::to_string(placed + presolve_depth + gpu_presolve_depth)
                  << " -D PRE_DEPTH=" << std::to_string(gpu_presolve_depth)
                  << " -D PRE_EXPANSION=" << std::to_string(gpu_presolve_expansion);
    std::string options = optionsStream.str();

    std::cout << "OPTIONS: " << options << std::endl;

    cl_int builderr = program.build(options.c_str());

    cl_int err = 0;
    std::cout << "OpenCL build log:" << std::endl;
    auto buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device, &err);
    std::cout << buildlog << std::endl;
    if(err != CL_SUCCESS) {
        std::cout << "getBuildInfo<CL_PROGRAM_BUILD_LOG> failed" << std::endl;
    }
    if(builderr != CL_SUCCESS) {
        std::cout << "program.build failed: " << builderr << std::endl;
        return false;
    }

    return true;
}

// should be a multiple of 64 at least for AMD GPUs
// ideally would be CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE
// bigger BATCH_SIZE means higher memory usage
constexpr size_t BATCH_SIZE = WORKGROUP_SIZE*(1 << 10);
typedef cl_uint result_type;

typedef struct {
    std::vector<result_type> hostOutputBuf;
    cl::Buffer clStartBuf;
    cl::Buffer clInterBuf;
    cl::Buffer clOutputBuf;
    cl::Kernel clKernel;
    cl::Kernel clPreKernel;
} batch;

uint64_t ClSolver::threadWorker()
{
    cl_int err = CL_SUCCESS;
    auto pre = nextPre();

    if(pre.empty()) {
        return 0;
    }

    // buffer
    batch b;

    // Create command queue.
    cl::CommandQueue cmdQueue = cl::CommandQueue(context, device, 0, &err);
    if(err != CL_SUCCESS) {
        std::cout << "failed to create command queue: " << err << std::endl;
        return 0;
    }

    // create device kernel
    b.clKernel = cl::Kernel(program, "solve_subboard", &err);
    if(err != CL_SUCCESS) {
        std::cout << "cl::Kernel failed: " << err << std::endl;
    }

    // create device presolver kernel
    b.clPreKernel = cl::Kernel(program, "pre_solve_subboard", &err);
    if(err != CL_SUCCESS) {
        std::cout << "cl::PreKernel failed: " << err << std::endl;
    }

    // Allocate start condition buffer on device
    b.clStartBuf = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
        BATCH_SIZE * sizeof(start_condition), nullptr, &err);
    if(err != CL_SUCCESS) {
        std::cout << "cl::Buffer start_buf failed: " << err << std::endl;
    }

    // Allocate intermediate buffer on device
    b.clInterBuf = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
        BATCH_SIZE * sizeof(start_condition) * gpu_presolve_expansion, nullptr, &err);
    if(err != CL_SUCCESS) {
        std::cout << "cl::Buffer start_buf failed: " << err << std::endl;
    }

    // TODO(sudden6): not the most efficient way to create a zero buffer on the device
    // Allocate result buffer on host
    b.hostOutputBuf = std::vector<result_type>(BATCH_SIZE, 0);

#if 0
    // Needs OpenCL 1.2
    // Allocate result buffer on device
    b.clOutputBuf = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY,
        BATCH_SIZE * sizeof(result_type), nullptr, &err);
    if(err != CL_SUCCESS) {
        std::cout << "cl::Buffer results_buf failed: " << err << std::endl;
    }

    // zero the result buffer
    result_type pattern = 0;
    err = cmdQueue.enqueueFillBuffer(b.clOutputBuf, pattern,
                                     0, BATCH_SIZE * sizeof(result_type),
                                     nullptr, nullptr);
    if(err != CL_SUCCESS) {
        std::cout << "fillBuffer results_buf failed: " << err << std::endl;
    }
#else
    // Allocate result buffer on device
    b.clOutputBuf = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        BATCH_SIZE * sizeof(result_type), b.hostOutputBuf.data(), &err);
    if(err != CL_SUCCESS) {
        std::cout << "cl::Buffer results_buf failed: " << err << std::endl;
    }
#endif

    // Set prekernel parameters.
    err = b.clPreKernel.setArg(0, b.clStartBuf);
    if(err != CL_SUCCESS) {
        std::cout << "presolve_subboard.setArg(0) failed: " << err << std::endl;
    }

    err = b.clPreKernel.setArg(1, b.clInterBuf);
    if(err != CL_SUCCESS) {
        std::cout << "presolve_subboard.setArg(1) failed: " << err << std::endl;
    }

    // Set kernel parameters.
    err = b.clKernel.setArg(0, b.clInterBuf);
    if(err != CL_SUCCESS) {
        std::cout << "solve_subboard.setArg(0) failed: " << err << std::endl;
    }

    err = b.clKernel.setArg(1, b.clOutputBuf);
    if(err != CL_SUCCESS) {
        std::cout << "solve_subboard.setArg(1) failed: " << err << std::endl;
    }

    auto start_time = std::time(nullptr);

    while (!pre.empty()) {
        void* clStart = cmdQueue.enqueueMapBuffer(b.clStartBuf, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION,
                                                  0, BATCH_SIZE * sizeof(start_condition), nullptr, nullptr, &err);
        if(err != CL_SUCCESS) {
            std::cout << "Failed to map start buffer: " << err << std::endl;
        }

        if(!clStart) {
            std::cout << "Unexpected NULL pointer" << std::endl;
        }

        start_condition_t* curIt = static_cast<start_condition_t*>(clStart);
        const start_condition* startIt = curIt;
        const start_condition_t* endIt = curIt + BATCH_SIZE;
        // TODO(sudden6): cleanup
        while(curIt < endIt) {
            curIt = pre.getNext(curIt, endIt);
            if(pre.empty()) {
                auto end_time = std::time(nullptr);
                //std::cout << "pre block took " << difftime(end_time, start_time) << "s" << std::endl;
                start_time = end_time;
                pre = nextPre();
                if(pre.empty()) {
                    break;
                }
            }
        }

        const size_t batchSize = curIt - startIt;

        size_t rest = batchSize % WORKGROUP_SIZE;
        size_t whole = batchSize - rest;

        err = cmdQueue.enqueueUnmapMemObject(b.clStartBuf, clStart, nullptr, nullptr);
        if(err != CL_SUCCESS) {
            std::cout << "Failed to unmap start buffer: " << err << std::endl;
        }

        if(whole > 0) {
            // Launch pre kernel on the compute device.
            err = cmdQueue.enqueueNDRangeKernel(b.clPreKernel, cl::NullRange,
                                                cl::NDRange{whole}, cl::NDRange{WORKGROUP_SIZE},
                                                nullptr, nullptr);
            if(err != CL_SUCCESS) {
                std::cout << "[PRE] enqueueNDRangeKernel failed: " << err << std::endl;
            }

            // Launch kernel on the compute device.
            err = cmdQueue.enqueueNDRangeKernel(b.clKernel, cl::NullRange,
                                                cl::NDRange{whole}, cl::NDRange{WORKGROUP_SIZE},
                                                nullptr, nullptr);
            if(err != CL_SUCCESS) {
                std::cout << "enqueueNDRangeKernel failed: " << err << std::endl;
            }
        }

        if(rest > 0) {
            // Launch pre kernel on the compute device.
            err = cmdQueue.enqueueNDRangeKernel(b.clPreKernel, cl::NDRange{whole},
                                                cl::NDRange{rest}, cl::NDRange{1},
                                                nullptr, nullptr);
            if(err != CL_SUCCESS) {
                std::cout << "[PRE] enqueueNDRangeKernel failed: " << err << std::endl;
            }

            // Launch kernel on the compute device.
            err = cmdQueue.enqueueNDRangeKernel(b.clKernel, cl::NDRange{whole},
                                                cl::NDRange{rest}, cl::NDRange{1},
                                                nullptr, nullptr);
            if(err != CL_SUCCESS) {
                std::cout << "enqueueNDRangeKernel failed: " << err << std::endl;
            }
        }
    }

    err = cmdQueue.enqueueReadBuffer(b.clOutputBuf, CL_TRUE, 0,
                                     BATCH_SIZE * sizeof(result_type), b.hostOutputBuf.data(),
                                     nullptr, nullptr);
    if(err != CL_SUCCESS) {
        std::cout << "enqueueReadBuffer failed: " << err << std::endl;
    }

    uint64_t result = 0;

    // get data from completed batch
    for(size_t i = 0; i < BATCH_SIZE; i++) {
        result += b.hostOutputBuf[i];
    }

    return result;
}

PreSolver ClSolver::nextPre()
{
    auto result = PreSolver();
    size_t old_solved;
#pragma omp atomic capture
    { old_solved = solved; solved++; } // atomically update solved, but capture original value in old_solved
    if(old_solved < start.size()) {
        if(old_solved % 10 == 0) {
            #pragma omp critical
            {
                //std::cout << "Solving: " << old_solved << "/" << start.size() << std::endl;
            }
        }
        result = PreSolver(boardsize, placed, presolve_depth, start[old_solved]);
    }

    return result;
}

uint64_t ClSolver::solve_subboard(const std::vector<start_condition> &start)
{
    this->start = start;
    solved = 0;

    if(start.empty()) {
        return 0;
    }

    std::cout << "Number of Threads: " << NUM_CMDQUEUES << std::endl;
    std::cout << "Buffer per Thread: " << BATCH_SIZE * (sizeof(start_condition)*(1 + gpu_presolve_expansion) + sizeof(cl_uint))/(1024*1024) << "MB" << std::endl;

    uint64_t result = 0;
    #pragma omp parallel for reduction(+ : result)
    for(size_t i = 0; i < NUM_CMDQUEUES; i++) {
        result = threadWorker();
    }

    return result * 2;
}

/*
 * @brief Enumerates all available OpenCL devices
 */
void ClSolver::enumerate_devices()
{
    cl_int err = 0;
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if(platforms.empty()) {
        std::cout << "No OpenCL platforms found" << std::endl;
        return;
    }

    for(size_t platform_idx = 0; platform_idx < platforms.size(); platform_idx++) {

        const auto& platform = platforms[platform_idx];

        const std::string platform_str = "Platform[" + std::to_string(platform_idx) + "]";

        std::cout << platform_str << " name: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
        std::cout << platform_str << " version: " << platform.getInfo<CL_PLATFORM_VERSION>() << std::endl;

        std::vector<cl::Device> devices;

        err = platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        if(err != CL_SUCCESS) {
            std::cout << "getDevices failed" << std::endl;
            continue;
        }

        for(size_t device_idx = 0; device_idx < devices.size(); device_idx++) {
            const std::string device_str = platform_str + " Device[" + std::to_string(device_idx) + "]";
            const auto& device = devices[device_idx];
            if (device.getInfo<CL_DEVICE_AVAILABLE>(&err)) {
                std::string name = device.getInfo<CL_DEVICE_NAME>(&err);
                if(err != CL_SUCCESS) {
                    name = "N/A";
                }

                std::string version = device.getInfo<CL_DEVICE_VERSION>();

                if(err != CL_SUCCESS) {
                    version = "N/A";
                }

                std::cout << device_str << " name: " << name << std::endl;
                std::cout << device_str << " version: " << version << std::endl;

            }
            if(err != CL_SUCCESS) {
                std::cout << "getInfo<CL_DEVICE_AVAILABLE> failed" << std::endl;
                continue;
            }
        }
    }
}

ClSolver* ClSolver::makeClSolver(unsigned int platform, unsigned int device)
{
    cl_int err = 0;
    ClSolver* solver = new ClSolver();
    if(!solver) {
        std::cout << "Failed to allocate memory" << std::endl;
        return nullptr;
    }

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if(platforms.empty()) {
        std::cout << "No OpenCL platforms found" << std::endl;
        return nullptr;
    }

    if(!(platform < platforms.size())) {
        std::cout << "Invalid OpenCL platform" << std::endl;
        return nullptr;
    }

    const cl::Platform& used_platform = platforms[platform];

    std::cout << "Platform name: " << used_platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
    std::cout << "Platform version: " << used_platform.getInfo<CL_PLATFORM_VERSION>() << std::endl;

    std::vector<cl::Device> devices;

    err = used_platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    if(err != CL_SUCCESS) {
        std::cout << "getDevices failed" << std::endl;
        return nullptr;
    }

    if(devices.empty()) {
        std::cout << "No devices found" << std::endl;
        return nullptr;
    }

    if(!(device < devices.size())) {
        std::cout << "Invalid OpenCL platform" << std::endl;
        return nullptr;
    }

    const cl::Device& used_device = devices[device];

    // check if device is available
    bool available = used_device.getInfo<CL_DEVICE_AVAILABLE>(&err);
    if(err != CL_SUCCESS) {
        std::cout << "getInfo<CL_DEVICE_AVAILABLE> failed" << std::endl;
        return nullptr;
    }

    if(!available) {
        std::cout << "OpenCL device not available" << std::endl;
        return nullptr;
    }

    solver->device = used_device;

    std::cout << "selected Device: " << used_device.getInfo<CL_DEVICE_NAME>(&err) << std::endl;
    if(err != CL_SUCCESS) {
        std::cout << "getInfo<CL_DEVICE_NAME> failed" << std::endl;
        return nullptr;
    }

    solver->context = cl::Context(used_device, nullptr, nullptr, nullptr, &err);
    if(err != CL_SUCCESS) {
        std::cout << "cl::Context failed" << std::endl;
        return nullptr;
    }

    // load source code
    std::ifstream sourcefile("clqueens.cl");
    std::string sourceStr((std::istreambuf_iterator<char>(sourcefile)),
                     std::istreambuf_iterator<char>());

    // create OpenCL program
    solver->program = cl::Program(solver->context, sourceStr, false, &err);
    if(err != CL_SUCCESS) {
        std::cout << "cl::Program failed" << std::endl;
        return nullptr;
    }

    return solver;
}



