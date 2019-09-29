#ifndef CPUSOLVER_H
#define CPUSOLVER_H

#include "isolver.h"
#include "parallel_hashmap/phmap.h"

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
    // depth 6 -> vec 316 <- seems to be the optimum for now
    // depth 7 -> vec 1100
    static constexpr uint8_t lookup_depth = 6;
    static constexpr size_t lut_vec_size = 316;
    static constexpr size_t max_candidates = 128;

    #pragma pack(push, 1)
    template <class T, size_t capacityA, size_t capacityB>
    class aligned_ABvec {
        static_assert(capacityA < UINT16_MAX);
        static_assert(capacityB < UINT16_MAX);

        T* begin;
        uint16_t Bfirst_empty;
        uint16_t Afirst_empty;
      public:
        aligned_ABvec()
        {
            const size_t total_capacity = capacityA + capacityB;
            begin = static_cast<T*>(aligned_alloc(16, total_capacity*sizeof(T)));
            Afirst_empty = 0;
            Bfirst_empty = 0;
        }

        aligned_ABvec (aligned_ABvec&& other) {
            this->begin = other.begin;
            this->Afirst_empty = other.Afirst_empty;
            this->Bfirst_empty = other.Bfirst_empty;
            other.begin = nullptr;
            other.Afirst_empty = 0;
            other.Bfirst_empty = 0;
        }

        aligned_ABvec(aligned_ABvec const&) = delete;
        aligned_ABvec& operator=(aligned_ABvec const&) = delete;

        ~aligned_ABvec()
        {
            free(begin);
            begin = nullptr;
        }

        bool valid()
        {
            return begin != nullptr;
        }

        size_t sizeA() const
        {
            return Afirst_empty;
        }

        size_t sizeB() const
        {
            return Bfirst_empty;
        }

        void clearA()
        {
            Afirst_empty = 0;
        }

        void clearB()
        {
            Bfirst_empty = 0;
        }

        const T*  dataA() const {
            return begin;
        }

        T* dataA() {
            return begin;
        }

        const T*  dataB() const {
            return begin + capacityA;
        }

        T*  dataB() {
            return begin + capacityA;
        }

        void push_backA(T& element)
        {
            assert(sizeA() < capacityA);
            dataA()[sizeA()] = element;
            Afirst_empty++;
        }

        void push_backB(T& element)
        {
            assert(sizeB() < capacityB);
            dataB()[sizeB()] = element;
            Bfirst_empty++;
        }
    };
    #pragma pack(pop)

#pragma pack(push, 1)
        typedef struct
        {
            uint32_t diagr;
            uint32_t diagl;
        } diags_packed_t;
#pragma pack(pop)

    uint_fast64_t stat_lookups = 0;
    uint_fast64_t stat_lookups_found = 0;
    uint_fast64_t stat_cmps = 0;

    phmap::flat_hash_map<uint32_t, aligned_ABvec<diags_packed_t, lut_vec_size, max_candidates>> lookup_hash;
    uint64_t get_solution_cnt(uint32_t cols, diags_packed_t search_elem);
    uint64_t count_solutions(const aligned_ABvec<diags_packed_t, lut_vec_size, max_candidates> &candidates);
    uint64_t count_solutions_fixed(const aligned_ABvec<diags_packed_t, lut_vec_size, max_candidates> &candidates);
};

#endif // CPUSOLVER_H
