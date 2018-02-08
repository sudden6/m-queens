#ifndef SOLVERSTRUCTS_H
#define SOLVERSTRUCTS_H

#include <cstdint>

typedef struct __attribute__ ((packed)) {
    uint32_t cols; // bitfield with all the used columns
    uint32_t diagl;// bitfield with all the used diagonals down left
    uint32_t diagr;// bitfield with all the used diagonals down right
    //uint32_t placed;// number of rows where queens are already placed
} start_condition;

#endif // SOLVERSTRUCTS_H
