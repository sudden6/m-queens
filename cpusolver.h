#ifndef CPUSOLVER_H
#define CPUSOLVER_H

#include "isolver.h"
#include <unordered_map>
#include <vector>

class cpuSolver : public ISolver
{
public:
    cpuSolver();
    bool init(uint8_t boardsize, uint8_t placed);
    uint64_t solve_subboard(const std::vector<start_condition>& starts);
    size_t init_lookup(uint8_t depth, uint32_t skip_mask);


    typedef struct
    {
        uint32_t diag_r;
        uint32_t diag_l;
    } lookup_t;

private:
    uint_fast8_t boardsize = 0;
    uint_fast8_t placed = 0;

    std::unordered_map<uint_fast32_t, std::vector<lookup_t>> lookup_hash;
};

#endif // CPUSOLVER_H
