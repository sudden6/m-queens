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

static constexpr size_t NUM_CMDQUEUES = 1;
constexpr uint_fast8_t GPU_DEPTH = 11;
constexpr size_t WORKGROUP_SIZE = 64;
constexpr size_t WORKSPACE_SIZE = 1024*1024*50;
constexpr size_t WORKSPACE_DEPTH = GPU_DEPTH - 1;

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
    gpu_depth = GPU_DEPTH;
    if((boardsize - placed) < GPU_DEPTH) {
        gpu_depth = boardsize - placed;
        presolve_depth = 0;
    } else {
        presolve_depth = boardsize - placed - GPU_DEPTH;
    }

    std::ostringstream optionsStream;
    optionsStream << "-cl-std=CL2.0"
                  << " -DBOARDSIZE=" << std::to_string(boardsize)
                  << " -DGPU_DEPTH=" <<std::to_string(gpu_depth)
                  << " -DWORKSPACE_SIZE=" <<std::to_string(WORKSPACE_SIZE)
                  << " -DWORKGROUP_SIZE=" <<std::to_string(WORKGROUP_SIZE);
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

    size_t workspace_mem = WORKSPACE_SIZE * WORKSPACE_DEPTH * sizeof(start_condition_t);
    size_t res_mem = WORKSPACE_SIZE * sizeof(cl_ulong);
    size_t dev_mem = workspace_mem + res_mem;

    std::cout << "OCL Kernel memory: " << std::to_string(dev_mem/(1024*1024)) << "MB" << std::endl;

    return true;
}

typedef cl_ulong result_type;

void ClSolver::threadWorker(uint32_t id, std::mutex &pre_lock)
{
    cl_int err = CL_SUCCESS;
    auto pre = nextPre(pre_lock);

    if(pre.empty()) {
        return;
    }

    // Create command queue.
    cl::DeviceCommandQueue devQueue = cl::DeviceCommandQueue::makeDefault(context, device, &err);
    if(err != CL_SUCCESS) {
        std::cout << "failed to create device command queue: " << err << std::endl;
        return;
    }
    
    cl::CommandQueue cmdQueue = cl::CommandQueue(context, device, cl::QueueProperties::None, &err);
    if(err != CL_SUCCESS) {
        std::cout << "failed to create command queue: " << err << std::endl;
        return;
    }

    // create device kernel
    cl::Kernel clKernel = cl::Kernel(program, "relaunch_kernel", &err);
    if(err != CL_SUCCESS) {
        std::cout << "cl::Kernel failed: " << err << std::endl;
    }

    // Allocate workspace buffer on device
    cl::Buffer clWorkspaceBuf = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_HOST_WRITE_ONLY,
        WORKSPACE_SIZE * WORKSPACE_DEPTH * sizeof(start_condition), nullptr, &err);
    if(err != CL_SUCCESS) {
        std::cout << "cl::Buffer start_buf failed: " << err << std::endl;
    }

    // Allocate workspace size buffer on device
    cl::Buffer clWorkspaceSizeBuf = cl::Buffer(context, CL_MEM_READ_WRITE,
        WORKSPACE_DEPTH * sizeof(cl_uint), nullptr, &err);
    if(err != CL_SUCCESS) {
        std::cout << "cl::Buffer start_buf failed: " << err << std::endl;
    }

    // host side is initially empty
    size_t hostStartFill = 0;

    // zero the workspace size buffer
    err = cmdQueue.enqueueFillBuffer(clWorkspaceSizeBuf, static_cast<cl_uint>(0),
                                     0, WORKSPACE_DEPTH * sizeof(cl_uint),
                                     nullptr, nullptr);
    if(err != CL_SUCCESS) {
        std::cout << "fillBuffer clWorkspaceSizeBuf failed: " << err << std::endl;
    }

    // Needs OpenCL 1.2
    // Allocate result buffer on device
    cl::Buffer clOutputBuf = cl::Buffer(context, CL_MEM_READ_WRITE,
        WORKSPACE_SIZE * sizeof(result_type), nullptr, &err);
    if(err != CL_SUCCESS) {
        std::cout << "cl::Buffer results_buf failed: " << err << std::endl;
    }

    // zero the result buffer
    result_type pattern = 0;
    err = cmdQueue.enqueueFillBuffer(clOutputBuf, pattern,
                                     0, WORKSPACE_SIZE * sizeof(result_type),
                                     nullptr, nullptr);
    if(err != CL_SUCCESS) {
        std::cout << "fillBuffer results_buf failed: " << err << std::endl;
    }

    // Set kernel parameters.
    err = clKernel.setArg(0, clWorkspaceBuf);
    if(err != CL_SUCCESS) {
        std::cout << "solve_subboard.setArg(0 failed: " << err << std::endl;
    }

    err = clKernel.setArg(1, clWorkspaceSizeBuf);
    if(err != CL_SUCCESS) {
        std::cout << "solve_subboard.setArg(1 failed: " << err << std::endl;
    }

    err = clKernel.setArg(2, clOutputBuf);
    if(err != CL_SUCCESS) {
        std::cout << "solve_subboard.setArg(2 failed: " << err << std::endl;
    }

    err = clKernel.setArg(3, static_cast<cl_uint>(CLSOLVER_FEED));
    if(err != CL_SUCCESS) {
        std::cout << "solve_subboard.setArg(3 failed: " << err << std::endl;
    }

    err = clKernel.setArg(4, static_cast<cl_uint>(0));
    if(err != CL_SUCCESS) {
        std::cout << "solve_subboard.setArg(4 failed: " << err << std::endl;
    }


    auto start_time = std::time(nullptr);

    bool finished = false;
    bool feeding = true;
    std::vector<start_condition_t> hostStartBuf(WORKSPACE_SIZE);
    std::vector<cl_uint> buffer_fill(WORKSPACE_DEPTH);
    while (!finished) {
        if (!pre.empty()) {
            auto curIt = hostStartBuf.begin() + hostStartFill;
            const auto endIt = hostStartBuf.cend();
            // TODO(sudden6): cleanup
            while(curIt < endIt) {
                curIt = pre.getNext(curIt, endIt);
                if(pre.empty()) {
                    auto end_time = std::time(nullptr);
                    //std::cout << "pre block took " << difftime(end_time, start_time) << "s" << std::endl;
                    start_time = end_time;
                    pre = nextPre(pre_lock);
                    if(pre.empty()) {
                        break;
                    }
                }
            }

            hostStartFill = std::distance(hostStartBuf.begin(), curIt);
        }

        err = cmdQueue.enqueueReadBuffer(clWorkspaceSizeBuf, CL_TRUE, 0, buffer_fill.size() * sizeof(cl_uint), buffer_fill.data());
        if (err != CL_SUCCESS) {
            std::cout << "enqueueReadBuffer clWorkspaceSizeBuf failed: " << err << std::endl;
            break;
        }

        const cl_uint first_fill = buffer_fill[0];

        bool workspace_empty = true;
        for(auto& fill: buffer_fill) {
            if(fill > 0) {
                workspace_empty = false;
                break;
            }
        }

        if ((hostStartFill == 0) && workspace_empty) {
            finished = true;
            break;
        } else if ((hostStartFill == 0) && feeding) {
            feeding = false;
            err = clKernel.setArg(3, static_cast<cl_uint>(CLSOLVER_CLEANUP));
            if(err != CL_SUCCESS) {
                std::cout << "solve_subboard.setArg(3 failed: " << err << std::endl;
            }

            hostStartBuf.clear();
            std::cout << "Starting cleanup" << std::endl;
        } else {
            const size_t first_free = WORKSPACE_SIZE - first_fill;
            const size_t batchSize = std::min(hostStartFill, first_free);
            hostStartFill -= batchSize;

            if (batchSize > 0) {
                const cl_uint new_buffer_fill = batchSize + first_fill;

                // write start conditions to workspace
                err = cmdQueue.enqueueWriteBuffer(clWorkspaceBuf, CL_TRUE,
                                                    first_fill * sizeof(start_condition), batchSize * sizeof(start_condition),
                                                    hostStartBuf.data() + hostStartFill, nullptr, nullptr);
                if(err != CL_SUCCESS) {
                    std::cout << "Failed to write start buffer: " << err << std::endl;
                }

                // write start condition count to workspace size
                err = cmdQueue.enqueueWriteBuffer(clWorkspaceSizeBuf, CL_TRUE,
                                                    0, sizeof(cl_uint),
                                                    &new_buffer_fill, nullptr, nullptr);
                if(err != CL_SUCCESS) {
                    std::cout << "Failed to write start buffer size: " << err << std::endl;
                }
            }
        }

        // Launch kernel on the compute device.
        err = cmdQueue.enqueueNDRangeKernel(clKernel, cl::NullRange,
                                            cl::NDRange{1}, cl::NDRange{1},
                                            nullptr, nullptr);
        if(err != CL_SUCCESS) {
            std::cout << "enqueueNDRangeKernel failed: " << err << std::endl;
        }
    }

    constexpr size_t SUM_REDUCTION_FACTOR = 1024*32;

    // create device kernel
    cl::Kernel sumKernel = cl::Kernel(program, "sum_results", &err);
    if(err != CL_SUCCESS) {
        std::cout << "cl::Kernel failed: " << err << std::endl;
    }

    cl::Buffer sumBuffer = cl::Buffer(context, CL_MEM_READ_WRITE,
                                      WORKSPACE_SIZE/SUM_REDUCTION_FACTOR * sizeof(cl_ulong), nullptr, &err);
    if(err != CL_SUCCESS) {
        std::cout << "cl::Buffer sumBuffer failed: " << err << std::endl;
    }

    // zero the sum buffer
    err = cmdQueue.enqueueFillBuffer(sumBuffer, static_cast<cl_ulong>(0),
                                     0,
                                     WORKSPACE_SIZE/SUM_REDUCTION_FACTOR * sizeof(cl_ulong),
                                     nullptr, nullptr);
    if(err != CL_SUCCESS) {
        std::cout << "fillBuffer clWorkspaceSizeBuf failed: " << err << std::endl;
    }

    err = sumKernel.setArg(0, clOutputBuf);
    if(err != CL_SUCCESS) {
        std::cout << "sumKernel.setArg(0 failed: " << err << std::endl;
    }

    sumKernel.setArg(1, sumBuffer);
    if(err != CL_SUCCESS) {
        std::cout << "sumKernel.setArg(1 failed: " << err << std::endl;
    }

    // Launch kernel to sum results
    err = cmdQueue.enqueueNDRangeKernel(sumKernel, cl::NullRange,
                                        cl::NDRange{WORKSPACE_SIZE/SUM_REDUCTION_FACTOR}, cl::NullRange,
                                        nullptr, nullptr);
    if(err != CL_SUCCESS) {
        std::cout << "enqueueNDRangeKernel(sumKernel) failed: " << err << std::endl;
    }

    std::vector<cl_ulong> sumHostBuffer(WORKSPACE_SIZE/SUM_REDUCTION_FACTOR);
    err = cmdQueue.enqueueReadBuffer(sumBuffer, CL_TRUE, 0, sumHostBuffer.size() * sizeof(cl_ulong), sumHostBuffer.data());
    if (err != CL_SUCCESS) {
        std::cout << "enqueueReadBuffer clWorkspaceSizeBuf failed: " << err << std::endl;
        return;
    }

    uint64_t result = 0;

    // get data from completed batch
    for(size_t i = 0; i < sumHostBuffer.size(); i++) {
        result += sumHostBuffer[i];
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
    std::cout << "Buffer per Thread: " << WORKSPACE_SIZE * sizeof(start_condition)/(1024*1024) << "MB" << std::endl;

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
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if(platforms.empty()) {
        std::cout << "No OpenCL platforms found" << std::endl;
        return nullptr;
    }

    if(!(platform < platforms.size())) {
        std::cout << "Invalid OpenCL platform index" << std::endl;
        return nullptr;
    }

    const cl::Platform& used_platform = platforms[platform];

    std::vector<cl::Device> devices;

    cl_int err = used_platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    if(err != CL_SUCCESS) {
        std::cout << "getDevices failed" << std::endl;
        return nullptr;
    }

    if(devices.empty()) {
        std::cout << "No devices found" << std::endl;
        return nullptr;
    }

    if(!(device < devices.size())) {
        std::cout << "Invalid OpenCL device index" << std::endl;
        return nullptr;
    }

    const cl::Device& used_device = devices[device];

    return makeClSolver(used_platform, used_device);
}

ClSolver* ClSolver::makeClSolver(cl::Platform platform, cl::Device used_device)
{
    cl_int err = 0;
    ClSolver* solver = new ClSolver();
    if(!solver) {
        std::cout << "Failed to allocate memory" << std::endl;
        return nullptr;
    }

    std::cout << "Platform name:    " << platform.getInfo<CL_PLATFORM_NAME>(&err) << std::endl;
    if(err != CL_SUCCESS) {
        std::cout << "getInfogetInfo<CL_PLATFORM_NAME> failed: " << std::to_string(err) << std::endl;
        return nullptr;
    }

    std::cout << "Platform version: " << platform.getInfo<CL_PLATFORM_VERSION>() << std::endl;
    if(err != CL_SUCCESS) {
        std::cout << "getInfogetInfo<CL_PLATFORM_VERSION> failed: " << std::to_string(err) << std::endl;
        return nullptr;
    }

    std::cout << "Device:           " << used_device.getInfo<CL_DEVICE_NAME>(&err) << std::endl;
    if(err != CL_SUCCESS) {
        std::cout << "getInfogetInfo<CL_DEVICE_NAME> failed: " << std::to_string(err) << std::endl;
        return nullptr;
    }

    std::string version_info = used_device.getInfo<CL_DEVICE_VERSION>(&err);
    std::cout << "Device version:   " << version_info << std::endl << std::endl;
    if(err != CL_SUCCESS) {
        std::cout << "getInfo<CL_DEVICE_VERSION> failed: " << std::to_string(err) << std::endl;
        return nullptr;
    }

    // See: https://www.khronos.org/registry/OpenCL/sdk/2.2/docs/man/html/clGetDeviceInfo.html
    const std::string min_version = "OpenCL 2.";
    if (version_info.compare(0, min_version.length(), min_version) != 0) {
        std::cout << "Not an OpenCL 2.x device, version: " << version_info << std::endl;
        return nullptr;
    }

    // check if device is available
    bool available = used_device.getInfo<CL_DEVICE_AVAILABLE>(&err);
    if(err != CL_SUCCESS) {
        std::cout << "getInfo<CL_DEVICE_AVAILABLE> failed: " << std::to_string(err) << std::endl;
        return nullptr;
    }

    if(!available) {
        std::cout << "OpenCL device not available" << std::endl;
        return nullptr;
    }

    solver->device = used_device;
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




