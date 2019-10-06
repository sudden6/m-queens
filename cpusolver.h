#ifndef CPUSOLVER_H
#define CPUSOLVER_H

#include "isolver.h"
#include "parallel_hashmap/phmap.h"
#include <vector>

static constexpr size_t AVX2_alignment = 32;

template <class T>
class aligned_vec {
    T* begin;
    T* first_empty;
  public:
    aligned_vec(size_t size, size_t init_size = 0)
        : begin{nullptr}
    {
        // handle zero size allocations
        if (size > 0) {
            begin = static_cast<T*>(aligned_alloc(AVX2_alignment, size*sizeof(T)));
            assert(begin);
        }
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
        assert(begin);
        return begin;
    }

    const T* data() const
    {
        assert(begin);
        return begin;
    }

    T& operator[] (size_t index)
    {
        assert(begin);
        assert(index < size());
        return begin[index];
    }

    const T& operator[] (size_t index) const
    {
        assert(begin);
        assert(index < size());
        return begin[index];
    }

    void push_back(T& element)
    {
        assert(begin);
        *first_empty = element;
        first_empty++;
    }
};

class cpuSolver : public ISolver
{
public:
    cpuSolver();
    bool init(uint8_t boardsize, uint8_t placed);
    uint64_t solve_subboard(const std::vector<start_condition_t>& starts);
    size_t init_lookup(uint8_t depth, uint32_t skip_mask);

private:
    uint_fast8_t boardsize = 0;
    uint_fast8_t placed = 0;

    static constexpr uint8_t max_lookup_depth = 6;
    static constexpr size_t max_candidates = 512;

    uint_fast64_t stat_lookups = 0;
    uint_fast64_t stat_lookups_not_found = 0;
    uint_fast64_t stat_cmps = 0;

    using lut_t = std::vector<aligned_vec<diags_packed_t>>;

    // maps column patterns to index in lookup_solutions;
    phmap::flat_hash_map<uint32_t, uint32_t> lookup_hash;
    // store solutions, this is constant after initializing the lookup table
    lut_t lookup_solutions;
    uint64_t get_solution_cnt(uint32_t cols, diags_packed_t search_elem, lut_t &lookup_candidates);
    uint64_t count_solutions(const aligned_vec<diags_packed_t> &solutions, const aligned_vec<diags_packed_t> &candidates);
    uint8_t lookup_depth(uint8_t boardsize, uint8_t placed);
};

#endif // CPUSOLVER_H
