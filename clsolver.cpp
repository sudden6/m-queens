#include "clsolver.h"
#include <ios>
#include <iostream>
#include <iterator>
#include <string>
#include <fstream>
#include <streambuf>
#include <sstream>

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

    err = platform.getDevices(CL_DEVICE_TYPE_CPU, &devices);
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
    std::ifstream sourcefile("clqueens.cl");
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

bool ClSolver::init(uint8_t boardsize)
{
    if(boardsize > MAXN || boardsize <= MINN) {
        std::cout << "Invalid boardsize for ClSolver" << std::endl;
        return false;
    }

    std::ostringstream optionsStream;
    optionsStream << "-D N=" << std::to_string(boardsize);
    std::string options = optionsStream.str();

    cl_int builderr = program.build(options.c_str());
    if(builderr != CL_SUCCESS) {
        std::cout << "program.build failed: " << builderr << std::endl;
    }

    cl_int err = 0;
    std::cout << "OpenCL build log:" << std::endl;
    auto buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device, &err);
    std::cout << buildlog << std::endl;
    if(err != CL_SUCCESS) {
        std::cout << "getBuildInfo<CL_PROGRAM_BUILD_LOG> failed" << std::endl;
    }
    return err && builderr;
}

uint64_t ClSolver::solve_subboard(std::vector<start_condition>& start)
{
    if(start.empty()) {
        return 0;
    }

    cl_int err = 0;

    // Create command queue.
    cl::CommandQueue queue(context, device, 0, &err);
    if(err != CL_SUCCESS) {
        std::cout << "CommandQueue failed: " << err << std::endl;
    }

    cl::Kernel solve_subboard(program, "solve_subboard", &err);
    if(err != CL_SUCCESS) {
        std::cout << "cl::Kernel failed: " << err << std::endl;
    }

    std::vector<cl_ulong> results(start.size(), 0);

    // Allocate device buffers and transfer input data to device.
    cl::Buffer start_buf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        start.size() * sizeof(start_condition), start.data(), &err);
    if(err != CL_SUCCESS) {
        std::cout << "cl::Buffer start_buf failed: " << err << std::endl;
    }

    cl::Buffer results_buf(context, CL_MEM_WRITE_ONLY,
        results.size() * sizeof(cl_ulong), nullptr, &err);
    if(err != CL_SUCCESS) {
        std::cout << "cl::Buffer results_buf failed: " << err << std::endl;
    }

    // Set kernel parameters.
    err = solve_subboard.setArg(0, start_buf);
    if(err != CL_SUCCESS) {
        std::cout << "solve_subboard.setArg(0 failed: " << err << std::endl;
    }

    err = solve_subboard.setArg(1, results_buf);
    if(err != CL_SUCCESS) {
        std::cout << "solve_subboard.setArg(1 failed: " << err << std::endl;
    }

    // Launch kernel on the compute device.
    err = queue.enqueueNDRangeKernel(solve_subboard, cl::NullRange, start.size(),
                                     cl::NullRange);
    if(err != CL_SUCCESS) {
        std::cout << "enqueueNDRangeKernel failed: " << err << std::endl;
    }

    // Get result back to host.
    err = queue.enqueueReadBuffer(results_buf, CL_TRUE, 0, results.size() * sizeof(cl_ulong), results.data());
    if(err != CL_SUCCESS) {
        std::cout << "enqueueReadBuffer failed: " << err << std::endl;
    }

    queue.finish();

    for(auto a : results) {
        std::cout << "test: " << a << std::endl;
    }
}



