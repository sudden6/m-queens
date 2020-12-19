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
constexpr size_t WORKSPACE_DEPTH = GPU_DEPTH - 1;
constexpr size_t SUM_REDUCTION_FACTOR = 1024*32;

bool ClSolver::init(uint8_t boardsize, uint8_t placed)
{
    if(boardsize > MAXN || boardsize < MINN) {
        std::cerr << "Invalid boardsize for ClSolver" << std::endl;
        return false;
    }

    if(placed >= boardsize) {
        std::cerr << "Invalid number of placed queens for ClSolver" << std::endl;
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
                  << " -DWORKSPACE_SIZE=" <<std::to_string(workspace_size)
                  << " -DWORKGROUP_SIZE=" <<std::to_string(WORKGROUP_SIZE);
    std::string options = optionsStream.str();

    std::cerr << "OPTIONS: " << options << std::endl;

    cl_int builderr = program.build(options.c_str());

    cl_int err = 0;
    std::cerr << "OpenCL build log:" << std::endl;
    auto buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device, &err);
    std::cerr << buildlog << std::endl;
    if(err != CL_SUCCESS) {
        std::cerr << "getBuildInfo<CL_PROGRAM_BUILD_LOG> failed" << std::endl;
    }
    if(builderr != CL_SUCCESS) {
        std::cerr << "program.build failed: " << builderr << std::endl;
        return false;
    }

    size_t workspace_mem = workspace_size * WORKSPACE_DEPTH * sizeof(start_condition_t);
    size_t res_mem = workspace_size * sizeof(cl_ulong);
    size_t sum_mem = workspace_size/SUM_REDUCTION_FACTOR * sizeof(cl_ulong);
    size_t dev_mem = workspace_mem + res_mem + sum_mem;

    std::cerr << "OCL Kernel memory: " << std::to_string(dev_mem/(1024*1024)) << "MB" << std::endl;
    std::cerr << "Threads: " << std::to_string(NUM_CMDQUEUES) << std::endl;

    if(!allocateThreads(NUM_CMDQUEUES)) {
        std::cerr << "Failed to allocate resource";
        return false;
    }

    return true;
}

typedef cl_ulong result_type;

bool ClSolver::allocateThreads(size_t cnt) {
    // ensure all resources are freed
    threads.clear();

    threads.resize(cnt);
    cl_int err = CL_SUCCESS;

    for(ThreadData& t: threads) {
        t.cmdQueue = cl::CommandQueue(context, device, cl::QueueProperties::None, &err);
        if(err != CL_SUCCESS) {
            std::cerr << "failed to create command queue: " << err << std::endl;
            return false;
        }

        // create device kernel
        t.clRelaunchKernel = cl::Kernel(program, "relaunch_kernel", &err);
        if(err != CL_SUCCESS) {
            std::cerr << "cl::Kernel failed: " << err << std::endl;
            return false;
        }

        // Allocate workspace buffer on device
        t.clWorkspaceBuf = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_HOST_WRITE_ONLY,
            workspace_size * WORKSPACE_DEPTH * sizeof(start_condition), nullptr, &err);
        if(err != CL_SUCCESS) {
            std::cerr << "cl::Buffer start_buf failed: " << err << std::endl;
            return false;
        }

        // Allocate workspace size buffer on device
        t.clWorkspaceSizeBuf = cl::Buffer(context, CL_MEM_READ_WRITE,
            WORKSPACE_DEPTH * sizeof(cl_uint), nullptr, &err);
        if(err != CL_SUCCESS) {
            std::cerr << "cl::Buffer start_buf failed: " << err << std::endl;
            return false;
        }

        // Allocate result buffer on device
        t.clOutputBuf = cl::Buffer(context, CL_MEM_READ_WRITE,
            workspace_size * sizeof(result_type), nullptr, &err);
        if(err != CL_SUCCESS) {
            std::cerr << "cl::Buffer results_buf failed: " << err << std::endl;
            return false;
        }

        // Set kernel parameters.
        err = t.clRelaunchKernel.setArg(0, t.clWorkspaceBuf);
        if(err != CL_SUCCESS) {
            std::cerr << "solve_subboard.setArg(0 failed: " << err << std::endl;
            return false;
        }

        err = t.clRelaunchKernel.setArg(1, t.clWorkspaceSizeBuf);
        if(err != CL_SUCCESS) {
            std::cerr << "solve_subboard.setArg(1 failed: " << err << std::endl;
            return false;
        }

        err = t.clRelaunchKernel.setArg(2, t.clOutputBuf);
        if(err != CL_SUCCESS) {
            std::cerr << "solve_subboard.setArg(2 failed: " << err << std::endl;
            return false;
        }

        err = t.clRelaunchKernel.setArg(3, static_cast<cl_uint>(CLSOLVER_FEED));
        if(err != CL_SUCCESS) {
            std::cerr << "solve_subboard.setArg(3 failed: " << err << std::endl;
            return false;
        }

        err = t.clRelaunchKernel.setArg(4, static_cast<cl_uint>(0));
        if(err != CL_SUCCESS) {
            std::cerr << "solve_subboard.setArg(4 failed: " << err << std::endl;
            return false;
        }

        // create device kernel
        t.sumKernel = cl::Kernel(program, "sum_results", &err);
        if(err != CL_SUCCESS) {
            std::cerr << "cl::Kernel failed: " << err << std::endl;
            return false;
        }

        t.sumBuffer = cl::Buffer(context, CL_MEM_READ_WRITE,
                                          workspace_size/SUM_REDUCTION_FACTOR * sizeof(cl_ulong), nullptr, &err);
        if(err != CL_SUCCESS) {
            std::cerr << "cl::Buffer sumBuffer failed: " << err << std::endl;
            return false;
        }

        err = t.sumKernel.setArg(0, t.clOutputBuf);
        if(err != CL_SUCCESS) {
            std::cerr << "sumKernel.setArg(0 failed: " << err << std::endl;
            return false;
        }

        t.sumKernel.setArg(1, t.sumBuffer);
        if(err != CL_SUCCESS) {
            std::cerr << "sumKernel.setArg(1 failed: " << err << std::endl;
            return false;
        }

        t.hostStartBuf.resize(workspace_size);
    }

    return true;
}

void ClSolver::threadWorker(uint32_t id, std::mutex &pre_lock)
{
    cl_int err = CL_SUCCESS;
    auto pre = nextPre(pre_lock);

    if(pre.empty()) {
        return;
    }

    ThreadData& t = threads[id];

    // host side is initially empty
    size_t hostStartFill = 0;

    // zero the workspace size buffer
    err = t.cmdQueue.enqueueFillBuffer(t.clWorkspaceSizeBuf, static_cast<cl_uint>(0),
                                     0, WORKSPACE_DEPTH * sizeof(cl_uint),
                                     nullptr, nullptr);
    if(err != CL_SUCCESS) {
        std::cerr << "fillBuffer clWorkspaceSizeBuf failed: " << err << std::endl;
    }

    // zero the result buffer
    result_type pattern = 0;
    err = t.cmdQueue.enqueueFillBuffer(t.clOutputBuf, pattern,
                                     0, workspace_size * sizeof(result_type),
                                     nullptr, nullptr);
    if(err != CL_SUCCESS) {
        std::cerr << "fillBuffer results_buf failed: " << err << std::endl;
    }

    // ensure we're starting with the correct state
    err = t.clRelaunchKernel.setArg(3, static_cast<cl_uint>(CLSOLVER_FEED));
    if(err != CL_SUCCESS) {
        std::cerr << "solve_subboard.setArg(3 failed: " << err << std::endl;
    }

    auto start_time = std::time(nullptr);

    bool finished = false;
    bool feeding = true;
    std::vector<cl_uint> buffer_fill(WORKSPACE_DEPTH);
    while (!finished) {
        if (!pre.empty()) {
            auto curIt = t.hostStartBuf.begin() + hostStartFill;
            const auto endIt = t.hostStartBuf.cend();
            // TODO(sudden6): cleanup
            while(curIt < endIt) {
                curIt = pre.getNext(curIt, endIt);
                if(pre.empty()) {
                    auto end_time = std::time(nullptr);
                    //std::cerr << "pre block took " << difftime(end_time, start_time) << "s" << std::endl;
                    start_time = end_time;
                    pre = nextPre(pre_lock);
                    if(pre.empty()) {
                        break;
                    }
                }
            }

            hostStartFill = std::distance(t.hostStartBuf.begin(), curIt);
        }

        err = t.cmdQueue.enqueueReadBuffer(t.clWorkspaceSizeBuf, CL_TRUE, 0, buffer_fill.size() * sizeof(cl_uint), buffer_fill.data());
        if (err != CL_SUCCESS) {
            std::cerr << "enqueueReadBuffer clWorkspaceSizeBuf failed: " << err << std::endl;
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
            err = t.clRelaunchKernel.setArg(3, static_cast<cl_uint>(CLSOLVER_CLEANUP));
            if(err != CL_SUCCESS) {
                std::cerr << "solve_subboard.setArg(3 failed: " << err << std::endl;
            }

            std::cerr << "Starting cleanup" << std::endl;
        } else {
            const size_t first_free = workspace_size - first_fill;
            const size_t batchSize = std::min(hostStartFill, first_free);
            hostStartFill -= batchSize;

            if (batchSize > 0) {
                const cl_uint new_buffer_fill = batchSize + first_fill;

                // write start conditions to workspace
                err = t.cmdQueue.enqueueWriteBuffer(t.clWorkspaceBuf, CL_TRUE,
                                                    first_fill * sizeof(start_condition), batchSize * sizeof(start_condition),
                                                    t.hostStartBuf.data() + hostStartFill, nullptr, nullptr);
                if(err != CL_SUCCESS) {
                    std::cerr << "Failed to write start buffer: " << err << std::endl;
                }

                // write start condition count to workspace size
                err = t.cmdQueue.enqueueWriteBuffer(t.clWorkspaceSizeBuf, CL_TRUE,
                                                    0, sizeof(cl_uint),
                                                    &new_buffer_fill, nullptr, nullptr);
                if(err != CL_SUCCESS) {
                    std::cerr << "Failed to write start buffer size: " << err << std::endl;
                }
            }
        }

        // Launch kernel on the compute device.
        err = t.cmdQueue.enqueueNDRangeKernel(t.clRelaunchKernel, cl::NullRange,
                                            cl::NDRange{1}, cl::NDRange{1},
                                            nullptr, nullptr);
        if(err != CL_SUCCESS) {
            std::cerr << "enqueueNDRangeKernel failed: " << err << std::endl;
        }
    }


    // zero the sum buffer
    err = t.cmdQueue.enqueueFillBuffer(t.sumBuffer, static_cast<cl_ulong>(0),
                                     0,
                                     workspace_size/SUM_REDUCTION_FACTOR * sizeof(cl_ulong),
                                     nullptr, nullptr);
    if(err != CL_SUCCESS) {
        std::cerr << "fillBuffer clWorkspaceSizeBuf failed: " << err << std::endl;
    }


    // Launch kernel to sum results
    err = t.cmdQueue.enqueueNDRangeKernel(t.sumKernel, cl::NullRange,
                                        cl::NDRange{workspace_size/SUM_REDUCTION_FACTOR}, cl::NullRange,
                                        nullptr, nullptr);
    if(err != CL_SUCCESS) {
        std::cerr << "enqueueNDRangeKernel(sumKernel) failed: " << err << std::endl;
    }

    std::vector<cl_ulong> sumHostBuffer(workspace_size/SUM_REDUCTION_FACTOR);
    err = t.cmdQueue.enqueueReadBuffer(t.sumBuffer, CL_TRUE, 0, sumHostBuffer.size() * sizeof(cl_ulong), sumHostBuffer.data());
    if (err != CL_SUCCESS) {
        std::cerr << "enqueueReadBuffer clWorkspaceSizeBuf failed: " << err << std::endl;
        return;
    }

    t.result = 0;

    // get data from completed batch
    for(size_t i = 0; i < sumHostBuffer.size(); i++) {
        t.result += sumHostBuffer[i];
    }
}

PreSolver ClSolver::nextPre(std::mutex& pre_lock)
{
    auto result = PreSolver();
    std::lock_guard<std::mutex> guard(pre_lock);

    if(solved < start.size()) {
        //std::cerr << "Solving: " << solved << "/" << start.size() << std::endl;
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

    std::mutex pre_lock{};

    for(size_t i = 0; i < threads.size(); i++) {
        ThreadData& t = threads[i];
        t.thread = std::unique_ptr<std::thread>(new std::thread(&ClSolver::threadWorker, this, i, std::ref(pre_lock)));
    }

    uint64_t result = 0;
    for(size_t i = 0; i < threads.size(); i++) {
        threads[i].thread->join();
        threads[i].thread.reset();
        result += threads[i].result;
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
        std::cerr << "No OpenCL platforms found" << std::endl;
        return;
    }

    for(size_t platform_idx = 0; platform_idx < platforms.size(); platform_idx++) {

        const auto& platform = platforms[platform_idx];

        const std::string platform_str = "Platform[" + std::to_string(platform_idx) + "]";

        std::cerr << platform_str << " name: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
        std::cerr << platform_str << " version: " << platform.getInfo<CL_PLATFORM_VERSION>() << std::endl;

        std::vector<cl::Device> devices;

        err = platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        if(err != CL_SUCCESS) {
            std::cerr << "getDevices failed" << std::endl;
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

                std::cerr << device_str << " name: " << name << std::endl;
                std::cerr << device_str << " version: " << version << std::endl;

            }
            if(err != CL_SUCCESS) {
                std::cerr << "getInfo<CL_DEVICE_AVAILABLE> failed" << std::endl;
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
        std::cerr << "No OpenCL platforms found" << std::endl;
        return nullptr;
    }

    if(!(platform < platforms.size())) {
        std::cerr << "Invalid OpenCL platform index" << std::endl;
        return nullptr;
    }

    const cl::Platform& used_platform = platforms[platform];

    std::vector<cl::Device> devices;

    cl_int err = used_platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    if(err != CL_SUCCESS) {
        std::cerr << "getDevices failed" << std::endl;
        return nullptr;
    }

    if(devices.empty()) {
        std::cerr << "No devices found" << std::endl;
        return nullptr;
    }

    if(!(device < devices.size())) {
        std::cerr << "Invalid OpenCL device index" << std::endl;
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
        std::cerr << "Failed to allocate memory" << std::endl;
        return nullptr;
    }

    std::cerr << "Platform name:    " << platform.getInfo<CL_PLATFORM_NAME>(&err) << std::endl;
    if(err != CL_SUCCESS) {
        std::cerr << "getInfogetInfo<CL_PLATFORM_NAME> failed: " << std::to_string(err) << std::endl;
        return nullptr;
    }

    std::cerr << "Platform version: " << platform.getInfo<CL_PLATFORM_VERSION>() << std::endl;
    if(err != CL_SUCCESS) {
        std::cerr << "getInfogetInfo<CL_PLATFORM_VERSION> failed: " << std::to_string(err) << std::endl;
        return nullptr;
    }

    std::cerr << "Device:           " << used_device.getInfo<CL_DEVICE_NAME>(&err) << std::endl;
    if(err != CL_SUCCESS) {
        std::cerr << "getInfogetInfo<CL_DEVICE_NAME> failed: " << std::to_string(err) << std::endl;
        return nullptr;
    }

    std::string version_info = used_device.getInfo<CL_DEVICE_VERSION>(&err);
    std::cerr << "Device version:   " << version_info << std::endl;
    if(err != CL_SUCCESS) {
        std::cerr << "getInfo<CL_DEVICE_VERSION> failed: " << std::to_string(err) << std::endl;
        return nullptr;
    }

    // See: https://www.khronos.org/registry/OpenCL/sdk/2.2/docs/man/html/clGetDeviceInfo.html
    const std::string min_version = "OpenCL 2.";
    if (version_info.compare(0, min_version.length(), min_version) != 0) {
        std::cerr << "Not an OpenCL 2.x device, version: " << version_info << std::endl;
        return nullptr;
    }

    // check if device is available
    bool available = used_device.getInfo<CL_DEVICE_AVAILABLE>(&err);
    if(err != CL_SUCCESS) {
        std::cerr << "getInfo<CL_DEVICE_AVAILABLE> failed: " << std::to_string(err) << std::endl;
        return nullptr;
    }

    if(!available) {
        std::cerr << "OpenCL device not available" << std::endl;
        return nullptr;
    }

    solver->device = used_device;

    cl_ulong memory_size = used_device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>(&err);
    if(err != CL_SUCCESS) {
        std::cerr << "getInfo<CL_DEVICE_GLOBAL_MEM_SIZE> failed: " << std::to_string(err) << std::endl;
        return nullptr;
    }

    std::cerr << "Device memory:    " << std::to_string(memory_size/(1024*1024)) << "MB" << std::endl << std::endl;

    cl_ulong memory_size_gb = memory_size / (1024*1024*1024);
    if (memory_size_gb >= 8) {
        solver->workspace_size = 1024*1024*48;
    } else if (memory_size_gb >= 4) {
        solver->workspace_size = 1024*1024*20;
    } else if (memory_size_gb >= 2) {
        solver->workspace_size = 1024*1024*8;
    } else {
        std::cerr << "Not enough memory" << std::endl;
        return nullptr;
    }

    solver->context = cl::Context(used_device, nullptr, nullptr, nullptr, &err);
    if(err != CL_SUCCESS) {
        std::cerr << "cl::Context failed" << std::endl;
        return nullptr;
    }

    // Create command queue.
    solver->devQueue = cl::DeviceCommandQueue::makeDefault(solver->context, solver->device, &err);
    if(err != CL_SUCCESS) {
        std::cerr << "DeviceCommandQueue::makeDefault() failed: " << err << std::endl;
        return nullptr;
    }

#if BOINC_OCL_SOLVER == 1
    // Hack to easily hardcode the kernel source
    std::string sourceStr =
#include "clqueens.cl"
;
#else
    // load source code
    std::ifstream sourcefile("clqueens.cl");
    std::string sourceStr((std::istreambuf_iterator<char>(sourcefile)),
                     std::istreambuf_iterator<char>());
#endif


    // create OpenCL program
    solver->program = cl::Program(solver->context, sourceStr, false, &err);
    if(err != CL_SUCCESS) {
        std::cerr << "cl::Program failed" << std::endl;
        return nullptr;
    }

    return solver;
}




