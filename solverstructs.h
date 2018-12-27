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

#elif defined(__GNUC__)
struct __attribute__ ((packed)) start_condition_t {
    uint32_t cols; // bitfield with all the used columns
    uint32_t diagl;// bitfield with all the used diagonals down left
    uint32_t diagr;// bitfield with all the used diagonals down right
};

#endif

struct Record {
    uint64_t diag_up;
    uint64_t diag_down;
    uint32_t hor;
    uint32_t vert;
};

typedef struct start_condition_t start_condition;

#endif // SOLVERSTRUCTS_H
