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
constexpr uint_fast8_t GPU_DEPTH = 10;
constexpr size_t WORKGROUP_SIZE = 64;

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
                  << " -D PLACED=" <<std::to_string(boardsize - gpu_depth)
                  << " -D DEPTH=" <<std::to_string(GPU_DEPTH)
                  << " -D WG_SIZE=" <<std::to_string(WORKGROUP_SIZE);
    std::string options = optionsStream.str();

    std::cout << "OPTIONS: " << options << std::endl;

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
constexpr size_t BATCH_SIZE = WORKGROUP_SIZE*(1 << 13);
typedef cl_uint result_type;

typedef struct {
    cl::Kernel clKernel;
    cl::Buffer clStartBuf;
    cl::Buffer clOutputBuf;
    std::vector<result_type> hostOutputBuf;
} batch;

constexpr size_t NUM_BATCHES = 1;

void ClSolver::threadWorker(uint32_t id, std::mutex &pre_lock)
{
    cl_int err = CL_SUCCESS;
    auto pre = nextPre(pre_lock);

    if(pre.empty()) {
        return;
    }

    // buffer
    batch batches[NUM_BATCHES];
    cl::CommandQueue cmdQueue;
    // Create command queue.
    cmdQueue = cl::CommandQueue(context, device, 0, &err);
    if(err != CL_SUCCESS) {
        std::cout << "failed to create command queue: " << err << std::endl;
        return;
    }

    for(size_t i = 0; i < NUM_BATCHES; i++) {
        batch& b = batches[i];

        // create device kernel
        b.clKernel = cl::Kernel(program, "solve_subboard", &err);
        if(err != CL_SUCCESS) {
            std::cout << "cl::Kernel failed: " << err << std::endl;
        }

        // Allocate start condition buffer on device
        b.clStartBuf = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
            BATCH_SIZE * sizeof(start_condition), nullptr, &err);
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


    auto start_time = std::time(nullptr);

    size_t cur_batch = 0;

    while (!pre.empty()) {
        auto& b = batches[cur_batch];
        cur_batch++;
        cur_batch %= NUM_BATCHES;

        void* clStart = cmdQueue.enqueueMapBuffer(b.clStartBuf, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION,
                                                  0, BATCH_SIZE * sizeof(start_condition), nullptr, nullptr, &err);
        if(err != CL_SUCCESS) {
            std::cout << "Failed to map start buffer: " << err << std::endl;
        }

        if(!clStart) {
            std::cout << "Unexpected NULL pointer" << std::endl;
        }
        start_condition * curIt = static_cast<start_condition*>(clStart);
        const start_condition * beginIt = curIt;
        const start_condition * endIt = beginIt + BATCH_SIZE;

        // TODO(sudden6): cleanup
        while(curIt < endIt) {
            curIt = pre.getNext(curIt, endIt);
            if(pre.empty()) {
                auto end_time = std::time(nullptr);
                std::cout << "pre block took " << difftime(end_time, start_time) << "s" << std::endl;
                start_time = end_time;
                pre = nextPre(pre_lock);
                if(pre.empty()) {
                    break;
                }
            }
        }

        const size_t batchSize = curIt - beginIt;

        size_t rest = batchSize % WORKGROUP_SIZE;
        size_t whole = batchSize - rest;

        err = cmdQueue.enqueueUnmapMemObject(b.clStartBuf, clStart, nullptr, nullptr);
        if(err != CL_SUCCESS) {
            std::cout << "Failed to unmap start buffer: " << err << std::endl;
        }

        if(whole > 0) {
            // Launch kernel on the compute device.
            err = cmdQueue.enqueueNDRangeKernel(b.clKernel, cl::NullRange,
                                                cl::NDRange{whole}, cl::NDRange{WORKGROUP_SIZE},
                                                nullptr, nullptr);
            if(err != CL_SUCCESS) {
                std::cout << "enqueueNDRangeKernel failed: " << err << std::endl;
            }
        }

        if(rest > 0) {
            // Launch kernel on the compute device.
            err = cmdQueue.enqueueNDRangeKernel(b.clKernel, cl::NDRange{whole},
                                                cl::NDRange{rest}, cl::NDRange{1},
                                                nullptr, nullptr);
            if(err != CL_SUCCESS) {
                std::cout << "enqueueNDRangeKernel failed: " << err << std::endl;
            }
        }
        // don't flood the device with commands
        cmdQueue.flush();
    }

    for(size_t i = 0; i < NUM_BATCHES; i++) {
        batch& b = batches[i];

        err = cmdQueue.enqueueReadBuffer(b.clOutputBuf, CL_FALSE, 0,
                                         BATCH_SIZE * sizeof(result_type), b.hostOutputBuf.data(),
                                         nullptr, nullptr);
        if(err != CL_SUCCESS) {
            std::cout << "enqueueReadBuffer failed: " << err << std::endl;
        }
    }

    uint64_t result = 0;

    cmdQueue.finish();

    for(size_t i = 0; i < NUM_BATCHES; i++) {
        batch& b = batches[i];

        // get data from completed batch
        for(size_t i = 0; i < BATCH_SIZE; i++) {
            result += b.hostOutputBuf[i];
        }
    }

    results[id] = result;
}

PreSolver ClSolver::nextPre(std::mutex& pre_lock)
{
    auto result = PreSolver();
    std::lock_guard<std::mutex> guard(pre_lock);

    if(solved < start.size()) {
        std::cout << "Solving: " << solved << "/" << start.size() << std::endl;
        result = PreSolver(boardsize, placed, presolve_depth, start[solved]);
        solved++;
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

    std::thread* threads[NUM_CMDQUEUES] = {nullptr};
    std::mutex pre_lock{};

    std::cout << "Number of Threads: " << NUM_CMDQUEUES << std::endl;
    std::cout << "Buffer per Thread: " << BATCH_SIZE * sizeof(start_condition)/(1024*1024) << "MB" << std::endl;

    for(size_t i = 0; i < NUM_CMDQUEUES; i++) {
        // init result
        results.push_back(0);

        threads[i] = new std::thread(&ClSolver::threadWorker, this, i, std::ref(pre_lock));
    }

    uint64_t result = 0;
    for(size_t i = 0; i < NUM_CMDQUEUES; i++) {
        threads[i]->join();
        result += results[i];
        delete threads[i];
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
    std::ifstream sourcefile("clqueens_amd.cl");
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



