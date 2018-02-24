#ifndef CLSOLVER_H
#define CLSOLVER_H

#include <cstdint>
#include <vector>
#include "solverstructs.h"

#include <CL/cl.hpp>

class ClSolver
{
public:
    ClSolver();
    bool init(uint8_t boardsize, uint8_t placed);
    uint64_t solve_subboard(const std::vector<start_condition>& start);
private:
    uint8_t presolve_depth = 0;
    uint8_t placed = 0;
    uint8_t boardsize = 0;
    cl::Context context;
    cl::Device device;
    cl::Program program;
    cl::CommandQueue cmdQueue;
};

#endif // CLSOLVER_H
