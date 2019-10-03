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

    template <class T>
    class aligned_vec {
        T* begin;
        T* first_empty;
      public:
        aligned_vec(size_t size, size_t init_size = 0)
        {
            begin = static_cast<T*>(aligned_alloc(AVX2_alignment, size*sizeof(T)));
            assert(begin);
            first_empty = begin + init_size;
        }

        aligned_vec (aligned_vec&& other) {
            this->begin = other.begin;
            this->first_empty = other.first_empty;
            other.begin = nullptr;
            other.first_empty = nullptr;
        }

        aligned_vec(aligned_vec const&) = delete;
        aligned_vec& operator=(aligned_vec const&) = delete;

        ~aligned_vec()
        {
            free(begin);
            begin = nullptr;
            first_empty = nullptr;
        }

        bool valid()
        {
            return begin != nullptr;
        }

        size_t size() const
        {
            return first_empty - begin;
        }

        void clear()
        {
            first_empty = begin;
        }

        T* data()
        {
            return begin;
        }

        const T* data() const
        {
            return begin;
        }

        T& operator[] (size_t index)
        {
            assert(index < size());
            return begin[index];
        }

        const T& operator[] (size_t index) const
        {
            assert(index < size());
            return begin[index];
        }

        void push_back(T& element)
        {
            *first_empty = element;
            first_empty++;
        }
    };

#pragma pack(push, 1)
        typedef struct alignas(8)
        {
            uint32_t diagr;
            uint32_t diagl;
        } diags_packed_t;
#pragma pack(pop)

private:
    uint_fast8_t boardsize = 0;
    uint_fast8_t placed = 0;
    // TODO(sudden6): find the maximum values that can be reached with this solver
    // depth 2 -> vec 2
    // depth 3 -> vec 6
    // depth 4 -> vec 24
    // depth 5 -> vec 88
    // depth 6 -> vec 552 for max N=27 <- seems to be the optimum for now
    // depth 7 -> vec 1100
    static constexpr uint8_t lookup_depth = 6;
    static constexpr size_t lut_vec_size = 552;
    static constexpr size_t max_candidates = 512;
    static constexpr size_t AVX2_alignment = 32;





    uint_fast64_t stat_lookups = 0;
    uint_fast64_t stat_lookups_found = 0;
    uint_fast64_t stat_cmps = 0;

    using lut_t = std::vector<aligned_vec<diags_packed_t>>;

    // maps column patterns to index in lookup_solutions;
    phmap::flat_hash_map<uint32_t, uint32_t> lookup_hash;
    // store solutions, this is constant after initializing the lookup table
    lut_t lookup_solutions;
    uint64_t get_solution_cnt(uint32_t cols, diags_packed_t search_elem, lut_t &lookup_candidates);
    uint64_t count_solutions(const aligned_vec<diags_packed_t> &solutions, const aligned_vec<diags_packed_t> &candidates);
    __attribute__ ((target ("default")))
    uint32_t count_solutions_fixed(const aligned_vec<diags_packed_t> &solutions, const aligned_vec<diags_packed_t> &candidates);
    __attribute__ ((target ("avx2")))
    uint32_t count_solutions_fixed(const aligned_vec<diags_packed_t>& solutions, const aligned_vec<diags_packed_t>& candidates);
    __attribute__ ((target ("sse4.2")))
    uint32_t count_solutions_fixed(const aligned_vec<diags_packed_t>& solutions, const aligned_vec<diags_packed_t>& candidates);
    __attribute__ ((target ("sse2")))
    uint32_t count_solutions_fixed(const aligned_vec<diags_packed_t>& solutions, const aligned_vec<diags_packed_t>& candidates);
};

#endif // CPUSOLVER_H
