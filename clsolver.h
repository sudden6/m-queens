#ifndef CLSOLVER_H
#define CLSOLVER_H

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
    ClSolver();
    bool init(uint8_t boardsize, uint8_t placed);
    uint64_t solve_subboard(const std::vector<start_condition>& start);
    static void enumerate_devices();

private:
    void threadWorker(uint32_t id, std::mutex &pre_lock);
    static constexpr size_t NUM_CMDQUEUES = 16;
    PreSolver nextPre(std::mutex &pre_lock);

    std::vector<start_condition> start;
    size_t solved = 0;

    uint8_t presolve_depth = 0;
    uint8_t placed = 0;
    uint8_t boardsize = 0;
    cl::Context context;
    cl::Device device;
    cl::Program program;
    std::vector<uint64_t> results;
};

#endif // CLSOLVER_H
