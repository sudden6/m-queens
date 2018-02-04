#ifndef CPUSOLVER_H
#define CPUSOLVER_H

#include <cstdint>
#include <vector>
#include "solverstructs.h"


class cpuSolver
{
public:
    cpuSolver();
    bool init(uint8_t boardsize);
    uint64_t solve_subboard(const std::vector<start_condition>& starts);
private:
    uint_fast8_t boardsize = 0;
};

#endif // CPUSOLVER_H
