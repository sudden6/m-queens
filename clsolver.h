#ifndef CLSOLVER_H
#define CLSOLVER_H

#include <atomic>
#include <cstdint>
#include <vector>
#include <mutex>
#include "solverstructs.h"
#include "presolver.h"
#include "isolver.h"
#include <CL/cl.h>
#include <CL/cl.hpp>

class ClSolver : public ISolver
{
public:
    bool init(uint8_t boardsize, uint8_t placed);
    uint64_t solve_subboard(const std::vector<start_condition>& start);
    static void enumerate_devices();

    static ClSolver* makeClSolver(unsigned int platform, unsigned int device);

private:
    ClSolver();
    static constexpr size_t NUM_CMDQUEUES = 8;
    PreSolver nextPre();
    uint64_t threadWorker();
    uint64_t expansion(uint8_t boardsize, uint8_t cur_idx, uint8_t depth);

    std::vector<start_condition> start;
    size_t solved = 0;

    uint8_t presolve_depth = 0;
    uint8_t gpu_presolve_depth = 0;
    uint8_t gpu_presolve_expansion = 0;
    uint8_t placed = 0;
    uint8_t boardsize = 0;
    cl::Context context;
    cl::Device device;
    cl::Program program;
};

#endif // CLSOLVER_H
