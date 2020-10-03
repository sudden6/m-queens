#include "claccell.h"

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

ClAccell::ClAccell()
{
}

ClAccell* ClAccell::makeClAccell(unsigned int platform, unsigned int device)
{
    cl_int err = 0;
    ClAccell* solver = new ClAccell();
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
    std::ifstream sourcefile("claccell.cl");
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

bool ClAccell::init(const cpuSolver::lut_t& lut_high_prob, const cpuSolver::lut_t& lut_low_prob)
{
    assert(lut_high_prob.size() == lut_low_prob.size());

    const size_t lut_size = lut_high_prob.size();

    lut_low_prob_sizes.resize(lut_size);
    lut_high_prob_sizes.resize(lut_size);

    // compute max size for padding
    for(size_t i = 0; i < lut_size; i++) {
        const size_t high_prob_size = lut_high_prob[i].size();
        lut_high_prob_max = std::max(lut_high_prob_max, high_prob_size);
        lut_high_prob_sizes[i] = high_prob_size;

        const size_t low_prob_size = lut_low_prob[i].size();
        lut_low_prob_max = std::max(lut_low_prob_max, low_prob_size);
        lut_low_prob_sizes[i] = low_prob_size;
    }

    // allocate flat array for luts
    std::vector<diags_packed_t> flat_low_prob;
    std::vector<diags_packed_t> flat_high_prob;

    flat_low_prob.resize(lut_low_prob_max*lut_size);
    flat_high_prob.resize(lut_high_prob_max*lut_size);

    // copy to flat array
    for(size_t i = 0; i < lut_size; i++) {
        diags_packed_t *high_prob_dst = flat_high_prob.data() + i*lut_high_prob_max;
        diags_packed_t *low_prob_dst = flat_low_prob.data() + i*lut_low_prob_max;
        size_t high_prob_len = lut_high_prob[i].size()*sizeof (diags_packed_t);
        size_t low_prob_len = lut_low_prob[i].size()*sizeof (diags_packed_t);

        memcpy(high_prob_dst, lut_high_prob[i].data(), high_prob_len);
        memcpy(low_prob_dst, lut_low_prob[i].data(), low_prob_len);
    }

    cl_int err = 0;

    // allocate OpenCL buffers for lut
    clFlatHighProb = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
                flat_high_prob.size() * sizeof(diags_packed_t), flat_high_prob.data(), &err);
    if(err != CL_SUCCESS) {
        std::cout << "cl::Buffer clFlatHighProb failed: " << err << std::endl;
        return false;
    }

    clFlatLowProb = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
                flat_low_prob.size() * sizeof(diags_packed_t), flat_low_prob.data(), &err);
    if(err != CL_SUCCESS) {
        std::cout << "cl::Buffer clFlatLowProb failed: " << err << std::endl;
        return false;
    }

    // allocate OpenCL buffer for result
    std::vector<uint32_t> resultCnt;
    resultCnt.resize(cpuSolver::max_candidates, 0);
    clResultCnt = cl::Buffer(context, CL_MEM_READ_WRITE|CL_MEM_HOST_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                             resultCnt.size() * sizeof (uint32_t), resultCnt.data(), &err);
    if(err != CL_SUCCESS) {
        std::cout << "cl::Buffer clResultCnt failed: " << err << std::endl;
        return false;
    }

    // allocate OpenCL buffer for candidates

    clCanBuff = cl::Buffer(context, CL_MEM_READ_ONLY|CL_MEM_HOST_WRITE_ONLY,
                             cpuSolver::max_candidates * sizeof (diags_packed_t), nullptr, &err);
    if(err != CL_SUCCESS) {
        std::cout << "cl::Buffer clCanBuff failed: " << err << std::endl;
        return false;
    }

    // compile Program
    std::ostringstream optionsStream;
    optionsStream << "-D MAX_CANDIDATES=" << std::to_string(cpuSolver::max_candidates);
    std::string options = optionsStream.str();

    std::cout << "OPTIONS: " << options << std::endl;

    cl_int builderr = program.build(options.c_str());

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

    // Create command queue.
    cmdQueue = cl::CommandQueue(context, device, 0, &err);
    if(err != CL_SUCCESS) {
        std::cout << "failed to create command queue: " << err << std::endl;
        return false;
    }

    // create device kernel
    clKernel = cl::Kernel(program, "count_solutions_trans", &err);
    if(err != CL_SUCCESS) {
        std::cout << "cl::Kernel failed: " << err << std::endl;
        return {};
    }

    // allocate fixed args
    err = clKernel.setArg(1, clCanBuff);
    if(err != CL_SUCCESS) {
        std::cout << "setArg 1 failed: " << err << std::endl;
        return {};
    }
    err = clKernel.setArg(2, clResultCnt);
    if(err != CL_SUCCESS) {
        std::cout << "setArg 2 failed: " << err << std::endl;
        return {};
    }

    return true;
}

uint64_t ClAccell::count(uint32_t lut_idx, const aligned_vec<diags_packed_t>& candidates, bool prob)
{
    const auto& lut_lens = prob ? lut_high_prob_sizes : lut_low_prob_sizes;
    if(lut_lens[lut_idx] == 0) {
        return 0;
    }

    const std::lock_guard<std::mutex> queue_guard(queue_lock);
    cl_int err = 0;
    err = cmdQueue.enqueueWriteBuffer(clCanBuff, CL_TRUE, 0, candidates.size() * sizeof (diags_packed_t), candidates.data());

    // allocate dynamic args
    err = clKernel.setArg(0, prob ? clFlatHighProb : clFlatLowProb);
    if(err != CL_SUCCESS) {
        std::cout << "setArg 0 failed: " << err << std::endl;
        return {};
    }

    uint32_t stride = prob ? lut_high_prob_max : lut_low_prob_max;

    err = clKernel.setArg(3, stride*lut_idx);
    if(err != CL_SUCCESS) {
        std::cout << "setArg 3 failed: " << err << std::endl;
        return {};
    }

    uint32_t range = prob ? lut_high_prob_sizes[lut_idx] : lut_low_prob_sizes[lut_idx];
    err = clKernel.setArg(4, range);
    if(err != CL_SUCCESS) {
        std::cout << "setArg 4 failed: " << err << std::endl;
        return {};
    }

    // Launch kernel on the compute device.
    err = cmdQueue.enqueueNDRangeKernel(clKernel, cl::NullRange,
                                        cl::NDRange{cpuSolver::max_candidates}, cl::NullRange);
    if(err != CL_SUCCESS) {
        std::cout << "enqueueNDRangeKernel failed: " << err << std::endl;
        return {};
    }

    // HACK: do this only on demand
    return 0;//get_cl_count();
}

uint64_t ClAccell::get_count()
{
    return get_cl_count();
}

uint64_t ClAccell::get_cl_count()
{
    std::vector<uint32_t> resultCnt;
    resultCnt.resize(cpuSolver::max_candidates, 0);

    cl_int err = 0;
    err = cmdQueue.enqueueReadBuffer(clResultCnt, CL_TRUE, 0, resultCnt.size() * sizeof (uint32_t), resultCnt.data());
    if(err != CL_SUCCESS) {
        std::cout << "enqueueReadBuffer failed: " << err << std::endl;
    }

    uint64_t res = 0;

    for(const auto& i: resultCnt) {
        res += i;
    }

    // allocate OpenCL buffer for result
    std::vector<uint32_t> emptyBuf;
    emptyBuf.resize(cpuSolver::max_candidates, 0);
    clResultCnt = cl::Buffer(context, CL_MEM_READ_WRITE|CL_MEM_HOST_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                             emptyBuf.size() * sizeof (uint32_t), emptyBuf.data(), &err);
    if(err != CL_SUCCESS) {
        std::cout << "cl::Buffer clResultCnt failed: " << err << std::endl;
        return {};
    }

    return res;
}


