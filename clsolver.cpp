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
constexpr uint_fast8_t GPU_DEPTH = 8;
constexpr size_t WORKGROUP_SIZE = 12;

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
constexpr size_t BATCH_SIZE = WORKGROUP_SIZE*(1 << 18);
typedef cl_uint result_type;

typedef struct {
    cl::Kernel clKernel;
    cl::Buffer clStartBuf;
    cl::Buffer clOutputBuf;
    cl::Event clBatchDone;
    std::vector<cl::Event> clStartKernel = {cl::Event()};
    std::vector<cl::Event> clTmpResult = {cl::Event()};
    std::vector<cl::Event> clReadResult = {cl::Event()};
    //std::vector<start_condition> hostStartBuf;
    std::vector<result_type> hostOutputBuf;
    size_t size = 0;
    size_t used = 0;
    cl::CommandQueue* cmdQueue;
} batch;

constexpr size_t NUM_BATCHES = 1;

void ClSolver::threadWorker(uint32_t id, std::mutex& pre_lock)
{
    cl_int err = CL_SUCCESS;

    // buffer
    batch batches[NUM_BATCHES];
    std::list<size_t> running;
    std::list<size_t> complete;

    for(size_t i = 0; i < NUM_BATCHES; i++) {
        batch& b = batches[i];
        // add this slot to the free queue
        complete.push_back(i);
        // create device kernel
        b.clKernel = cl::Kernel(program, "solve_subboard", &err);
        if(err != CL_SUCCESS) {
            std::cout << "cl::Kernel failed: " << err << std::endl;
        }
        // Allocate start condition buffer on host
        //b.hostStartBuf.resize(BATCH_SIZE);

        // Allocate start condition buffer on device
        b.clStartBuf = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
            BATCH_SIZE * sizeof(start_condition), nullptr, &err);
        if(err != CL_SUCCESS) {
            std::cout << "cl::Buffer start_buf failed: " << err << std::endl;
        }

        // TODO(sudden6): not the most efficient way to create a zero buffer on the device
        // Allocate result buffer on host
        b.hostOutputBuf = std::vector<result_type>(BATCH_SIZE, 0);

        // Allocate result buffer on device
        b.clOutputBuf = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            BATCH_SIZE * sizeof(result_type), b.hostOutputBuf.data(), &err);
        if(err != CL_SUCCESS) {
            std::cout << "cl::Buffer results_buf failed: " << err << std::endl;
        }

        // Set kernel parameters.
        err = b.clKernel.setArg(0, b.clStartBuf);
        if(err != CL_SUCCESS) {
            std::cout << "solve_subboard.setArg(0 failed: " << err << std::endl;
        }

        err = b.clKernel.setArg(1, b.clOutputBuf);
        if(err != CL_SUCCESS) {
            std::cout << "solve_subboard.setArg(1 failed: " << err << std::endl;
        }

        b.cmdQueue = &queues[id];
    }

    uint64_t result = 0;

    auto pre = nextPre(pre_lock);
    auto start_time = std::time(nullptr);

    while (!pre.empty()) {
        // move idx from free queue to busy queue
        size_t cur = complete.front();
        complete.pop_front();
        running.push_back(cur);
        auto& b = batches[cur];
        auto& cmdQueue = b.cmdQueue;

        void* clStart = cmdQueue->enqueueMapBuffer(b.clStartBuf, TRUE, CL_MAP_WRITE, 0, BATCH_SIZE);
        start_condition * curIt = static_cast<start_condition*>(clStart);
        const start_condition * beginIt = curIt;
        const start_condition * endIt = beginIt + BATCH_SIZE;

        // TODO(sudden6): cleanup
        //auto hostBufIt = b.hostStartBuf.begin();
        while(curIt != endIt) {
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
        b.size = curIt - beginIt;
        const auto& batchSize = b.size;
        b.used = std::max(batchSize, b.used);



        if(err != CL_SUCCESS) {
            std::cout << "Failed to transfer start buffer: " << err << std::endl;
        }

        size_t rest = batchSize % WORKGROUP_SIZE;
        size_t whole = batchSize - rest;

        // Transfer start buffer to device
        //err = cmdQueue->enqueueWriteBuffer(b.clStartBuf, CL_FALSE, 0,
        //                                  batchSize * sizeof(start_condition), b.hostStartBuf.data(),
        //                                 nullptr, &b.clStartKernel[0]);

        cmdQueue->enqueueUnmapMemObject(b.clStartBuf, clStart);

        if(whole > 0) {
            // Launch kernel on the compute device.
            err = cmdQueue->enqueueNDRangeKernel(b.clKernel, cl::NullRange,
                                                cl::NDRange{whole}, cl::NDRange{WORKGROUP_SIZE},
                                                nullptr, rest ? &b.clTmpResult[0] : &b.clReadResult[0]);
            if(err != CL_SUCCESS) {
                std::cout << "enqueueNDRangeKernel failed: " << err << std::endl;
            }
        }

        if(rest > 0) {
            // Launch kernel on the compute device.
            err = cmdQueue->enqueueNDRangeKernel(b.clKernel, cl::NDRange{whole},
                                                cl::NDRange{rest}, cl::NDRange{1},
                                                whole ? &b.clTmpResult : nullptr, &b.clReadResult[0]);
            if(err != CL_SUCCESS) {
                std::cout << "enqueueNDRangeKernel failed: " << err << std::endl;
            }
        }

        // wait for free slot
        auto it = running.begin();
        while (complete.empty()) {
            // loop endless over running batches
            if(it == running.end()) {
                it = running.begin();
                std::this_thread::sleep_for(std::chrono::milliseconds(80));
            }

            size_t idx = *it;
            batch& c = batches[idx];
            // check if batch finished
            if(c.clReadResult[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>() == CL_COMPLETE) {
                // remove from running batches
                running.erase(it);
                // add to completed batches
                complete.push_back(idx);
            }
        }
    }

    running.clear();
    for(size_t i = 0; i < NUM_BATCHES; i++) {
        batch& b = batches[i];
        auto& cmdQueue = b.cmdQueue;
        if(b.used == 0) {
            continue;
        }
        err = cmdQueue->enqueueReadBuffer(b.clOutputBuf, CL_FALSE, 0,
                                         b.used * sizeof(result_type), b.hostOutputBuf.data(),
                                         &b.clReadResult, &b.clBatchDone);
        if(err != CL_SUCCESS) {
            std::cout << "enqueueReadBuffer failed: " << err << std::endl;
        }
        running.push_back(i);
    }

    auto it = running.begin();
    while (!running.empty()) {
        // loop endless over running batches
        if(it == running.end()) {
            it = running.begin();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        size_t idx = *it;
        batch& c = batches[idx];
        // check if batch finished
        if(c.clBatchDone.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>() == CL_COMPLETE) {
            // get data from completed batch
            for(size_t i = 0; i < c.used; i++) {
                result += c.hostOutputBuf[i];
            }
            // remove from running batches
            it = running.erase(it);
        }
    }

    results[id] = result;
}

PreSolver ClSolver::nextPre(std::mutex& pre_lock)
{
    auto result = PreSolver();
    std::lock_guard<std::mutex> lock(pre_lock);
    if(solved < start.size()) {
        std::cout << "Solving: " << solved << "/" << start.size() << std::endl;
        result = PreSolver(boardsize, placed, presolve_depth, start[solved]);
        solved++;
    }

    return result;
}

uint64_t ClSolver::solve_subboard(const std::vector<start_condition> &start)
{
    cl_int err = CL_SUCCESS;

    this->start = start;
    solved = 0;

    if(start.empty()) {
        return 0;
    }

    std::thread* threads[NUM_CMDQUEUES] = {nullptr};
    std::mutex pre_lock;

    for(size_t i = 0; i < NUM_CMDQUEUES; i++) {
        auto& cmdQueue = queues[i];
        // Create command queue.
        cmdQueue = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
        if(err != CL_SUCCESS) {
            std::cout << "CommandQueue failed, probably out-of-order-exec not supported: " << err << std::endl;
            cmdQueue = cl::CommandQueue(context, device, 0, &err);
            if(err != CL_SUCCESS) {
                std::cout << "failed to create command queue: " << err << std::endl;
            }
        }

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



