#ifndef CLSOLVER_H
#define CLSOLVER_H

#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/cl2.hpp>

#include <cstdlib>
#include <vector>
#include <mutex>
#include "solverstructs.h"
#include "presolver.h"
#include "isolver.h"


class ClSolver : public ISolver
{
public:
    bool init(uint8_t boardsize, uint8_t placed);
    uint64_t solve_subboard(const std::vector<start_condition>& start);
    static void enumerate_devices();

    static ClSolver* makeClSolver(unsigned int platform, unsigned int device);

private:
    ClSolver();
    void threadWorker(uint32_t id, std::mutex &pre_lock);
    PreSolver nextPre(std::mutex &pre_lock);

    std::vector<start_condition> start;
    size_t solved = 0;

    uint8_t gpu_depth = 0;
    uint8_t presolve_depth = 0;
    uint8_t placed = 0;
    uint8_t boardsize = 0;
    cl::Context context;
    cl::Device device;
    cl::Program program;
    std::vector<uint64_t> results;
};

#endif // CLSOLVER_H
