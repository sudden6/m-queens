#ifndef CPUSOLVER_H
#define CPUSOLVER_H

#include <cstdint>
#include <vector>
#include "solverstructs.h"


class cpuSolver
{
public:
    cpuSolver();
    uint64_t solve_subboard(uint_fast8_t n, const std::vector<start_condition>& starts);
};

#endif // CPUSOLVER_H
