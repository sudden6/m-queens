#ifndef CPUSOLVER_H
#define CPUSOLVER_H

#include "isolver.h"
#include "parallel_hashmap/phmap.h"
#include <array>
#include <vector>

class u32_hasher {
public:
    std::size_t operator()(uint32_t const &p) const
    {
        uint32_t x = p;
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = (x >> 16) ^ x;
        return x;
    }
};



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
    // depth 5 -> vec 88
    // depth 6 -> vec 316
    static constexpr uint8_t lookup_depth = 4;
    static constexpr size_t lut_vec_size = 24;
    static constexpr size_t max_candidates = 512;

    template <class T, size_t capacity>
    class aligned_vec {
        T* begin;
        T* first_empty;
      public:
        aligned_vec()
        {
            begin = static_cast<T*>(aligned_alloc(16, capacity*sizeof(T)));
            first_empty = begin;
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
            assert(size() < capacity);
            *first_empty = element;
            first_empty++;
        }
    };




    phmap::flat_hash_map<uint32_t, aligned_vec<uint64_t, lut_vec_size>, u32_hasher> lookup_hash;
    phmap::flat_hash_map<uint32_t, aligned_vec<uint64_t, max_candidates>> lookup_candidates;
    uint64_t get_solution_cnt(uint32_t cols, uint32_t diagl, uint32_t diagr);
    uint64_t count_solutions(const aligned_vec<uint64_t, max_candidates> &candidates, const aligned_vec<uint64_t, lut_vec_size> &solutions);
    uint64_t count_solutions_fixed(const aligned_vec<uint64_t, max_candidates>& candidates, const aligned_vec<uint64_t, lut_vec_size> &solutions);
};

#endif // CPUSOLVER_H
