#ifndef CPUSOLVER_H
#define CPUSOLVER_H

#include "isolver.h"
#include "parallel_hashmap/phmap.h"
#include <array>
#include <vector>

class cpuSolver : public ISolver
{
public:
    cpuSolver();
    bool init(uint8_t boardsize, uint8_t placed);
    uint64_t solve_subboard(const std::vector<start_condition>& starts);
    size_t init_lookup(uint8_t depth, uint32_t skip_mask);

private:
    uint_fast8_t boardsize = 0;
    uint_fast8_t placed = 0;
    // depth 2 -> vec 2
    // depth 3 -> vec 6
    // depth 4 -> vec 24
    static constexpr size_t lut_vec_size = 6;

    phmap::flat_hash_map<uint32_t, std::array<uint64_t, lut_vec_size>> lookup_hash;
    uint64_t get_solution_cnt(uint32_t cols, uint32_t diagl, uint32_t diagr);
};

#endif // CPUSOLVER_H
