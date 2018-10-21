#ifndef ISOLVER_H
#define ISOLVER_H

#include <cstdint>
#include <vector>
#include "solverstructs.h"

class ISolver {
public:
    virtual bool init(uint8_t boardsize, uint8_t placed) = 0;
    virtual uint64_t solve_subboard(const std::vector<start_condition>& starts) = 0;
    virtual ~ISolver(){}
};

#endif // ISOLVER_H
