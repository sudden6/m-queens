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
    static constexpr uint8_t lookup_depth = 5;
    static constexpr size_t lut_vec_size = 88;
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

    #pragma pack(push, 1)
    template <class T, size_t capacityA, size_t capacityB>
    class aligned_ABvec {
        static_assert(capacityA < UINT16_MAX);
        static constexpr uint16_t Afirst_empty_mask = 0xF800;

        T* begin;
        uint16_t Bfirst_empty;
        uint16_t Afirst_empty;
      public:
        aligned_ABvec()
        {
            const size_t total_capacity = capacityA + capacityB;
            begin = static_cast<T*>(aligned_alloc(16, total_capacity*sizeof(T)));
            Afirst_empty = 0;
            Bfirst_empty = capacityA;
        }

        aligned_ABvec (aligned_ABvec&& other) {
            this->begin = other.begin;
            this->Afirst_empty = other.Afirst_empty;
            this->Bfirst_empty = other.Bfirst_empty;
            other.begin = nullptr;
            other.Afirst_empty = 0;
            other.Bfirst_empty = capacityA;
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
            return Bfirst_empty - capacityA;
        }

        void clearA()
        {
            Afirst_empty = 0;
        }

        void clearB()
        {
            Bfirst_empty = capacityA;
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

    phmap::flat_hash_map<uint32_t, aligned_ABvec<uint64_t, lut_vec_size, max_candidates>> lookup_hash;
    //phmap::flat_hash_map<uint32_t, aligned_vec<uint64_t, max_candidates>> lookup_candidates;
    uint64_t get_solution_cnt(uint32_t cols, uint32_t diagl, uint32_t diagr);
    uint64_t count_solutions(const aligned_ABvec<uint64_t, lut_vec_size, max_candidates> &candidates);
    uint64_t count_solutions_fixed(const aligned_ABvec<uint64_t, lut_vec_size, max_candidates>& candidates);
};

#endif // CPUSOLVER_H
