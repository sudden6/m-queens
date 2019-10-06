#ifndef SOLVERSTRUCTS_H
#define SOLVERSTRUCTS_H

#include <cstdint>

#pragma pack(push, 1)
    typedef struct {
        uint32_t cols; // bitfield with all the used columns
        uint32_t diagl;// bitfield with all the used diagonals down left
        uint32_t diagr;// bitfield with all the used diagonals down right
    } start_condition_t;
#pragma pack(pop)

#pragma pack(push, 1)
    typedef struct alignas(8)
    {
        uint32_t diagr;
        uint32_t diagl;
    } diags_packed_t;
#pragma pack(pop)

#endif // SOLVERSTRUCTS_H
