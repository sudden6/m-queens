#ifndef CLACCELL_H
#define CLACCELL_H

#include <cstdint>
#include <vector>
#include <CL/cl.h>
#include <CL/cl.hpp>
#include "cpusolver.h"

class ClAccell
{
public:
    ClAccell();

    static ClAccell *makeClAccell(unsigned int platform, unsigned int device);
    bool init(const cpuSolver::lut_t &lut_high_prob, const cpuSolver::lut_t &lut_low_prob);
    uint64_t count(uint32_t lut_idx, const aligned_vec<diags_packed_t> &candidates, bool prob);
    uint64_t get_count();

private:
    uint64_t get_cl_count();
    cl::Context context;
    cl::Device device;
    cl::Program program;
    cl::CommandQueue cmdQueue;
    cl::Kernel clKernel;

    cl::Buffer clFlatHighProb;
    cl::Buffer clFlatLowProb;
    cl::Buffer clResultCnt;

    cl::Buffer clCanBuff;
    size_t first_free = 0;
    std::mutex queue_lock;


    uint64_t cnt = 0;

    size_t lut_high_prob_max = 0;
    size_t lut_low_prob_max = 0;
    std::vector<uint32_t> lut_high_prob_sizes;
    std::vector<uint32_t> lut_low_prob_sizes;
};

#endif // CLACCELL_H
