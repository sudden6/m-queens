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
    bool init(uint8_t boardsize);
    uint64_t solve_subboard(std::vector<start_condition> &start);
private:
    cl::Context context;
    cl::Device device;
    cl::Program program;
};

#endif // CLSOLVER_H
