#ifndef CLSOLVER_H
#define CLSOLVER_H

#include <cstdint>
#include <vector>
#include "solverstructs.h"
#include <CL/cl2.hpp>

class ClSolver
{
public:
    ClSolver();
    uint64_t solve_subboard(uint_fast8_t n, std::vector<start_condition> &start);
private:
    cl::Program program;
};

#endif // CLSOLVER_H
