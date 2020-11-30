#ifndef SOLVERSTRUCTS_H
#define SOLVERSTRUCTS_H

#include <cstdint>

#ifdef _MSC_VER

#pragma pack(push,1)
struct start_condition_t {
    uint32_t cols; // bitfield with all the used columns
    uint32_t diagl;// bitfield with all the used diagonals down left
    uint32_t diagr;// bitfield with all the used diagonals down right
};

#pragma pack(pop)

#else
struct __attribute__ ((packed)) start_condition_t {
    uint32_t cols; // bitfield with all the used columns
    uint32_t diagl;// bitfield with all the used diagonals down left
    uint32_t diagr;// bitfield with all the used diagonals down right
};

#endif

typedef struct start_condition_t start_condition;

#define CLSOLVER_STATE_MASK (1U << 31)
#define CLSOLVER_FEED (0)
#define CLSOLVER_CLEANUP CLSOLVER_STATE_MASK

#endif // SOLVERSTRUCTS_H
