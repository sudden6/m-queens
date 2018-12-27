#ifndef CPUSOLVER_H
#define CPUSOLVER_H

#include "isolver.h"

class cpuSolver : public ISolver
{
public:
    cpuSolver();
    bool init(uint8_t boardsize, uint8_t placed);
    uint64_t solve_subboard(const std::vector<start_condition>& starts);
    uint64_t solve_subboard(const std::vector<Preplacement> &starts);
private:
    uint_fast8_t boardsize = 0;
    uint_fast8_t placed = 0;
};

#endif // CPUSOLVER_H
