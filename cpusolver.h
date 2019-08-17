#ifndef CPUSOLVER_H
#define CPUSOLVER_H

#include "isolver.h"
#include "parallel_hashmap/phmap.h"
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

    typedef struct
    {
        std::vector<uint64_t> v0;
        std::vector<uint64_t> v1;
        std::vector<uint64_t> v2;
    } bin_lookup_t;

private:
    uint_fast8_t boardsize = 0;
    uint_fast8_t placed = 0;

    phmap::flat_hash_map<uint32_t, bin_lookup_t> lookup_hash;
    uint64_t get_solution_cnt(uint32_t cols, uint32_t diagl, uint32_t diagr);
};

#endif // CPUSOLVER_H
