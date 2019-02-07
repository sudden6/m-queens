typedef struct __attribute__ ((packed)) {
    uint cols; // bitfield with all the used columns
    uint diagl;// bitfield with all the used diagonals down left
    uint diagr;// bitfield with all the used diagonals down right
} start_condition;

typedef char int8_t;
typedef char int_fast8_t;
typedef uchar uint_fast8_t;
typedef short int_fast16_t;
typedef ushort uint_fast16_t;
typedef int int_fast32_t;
typedef uint uint_fast32_t;
#define UINT_FAST32_MAX UINT_MAX

// for convenience
#define L (get_local_id(0))
#define G (get_global_id(0))

kernel void solve_subboard(__global const start_condition* in_starts, __global uint* out_cnt) {
    // counter for the number of solutions
    // sufficient until n=29
    uint num = 0;
//#define LOOKAHEAD 3

#define DIAGL(d) diagl[L][(d)]
#define DIAGR(d) diagr[L][(d)]
#define POSIBS(d) posibs[L][(d)]
#define COLS(d) cols[L][(d)]

    __local uint_fast32_t cols[WG_SIZE][DEPTH];
    __local uint_fast32_t posibs[WG_SIZE][DEPTH]; // Our backtracking 'stack'
    __local uint_fast32_t diagl[WG_SIZE][DEPTH], diagr[WG_SIZE][DEPTH];
#ifdef LOOKAHEAD
    __local int8_t rest[WG_SIZE][DEPTH]; // number of rows left
#endif
    int_fast16_t d = 0; // d is our depth in the backtrack stack
    // The UINT_FAST32_MAX here is used to fill all 'coloumn' bits after n ...
    COLS(d) = in_starts[G].cols;
    // This places the first two queens
    DIAGL(d) = in_starts[G].diagl;
    DIAGR(d) = in_starts[G].diagr;
    // we're allready two rows into the field here
#ifdef LOOKAHEAD
    rest[L][d] = N - LOOKAHEAD - PLACED;
#endif

    //printf("G: %d, L: %d, cols: %x, diagl: %x, diagr: %x\n", G, L, COLS(d), diagl[L][d], diagr[L][d]);

    //  The variable posib contains the bitmask of possibilities we still have
    //  to try in a given row ...
    POSIBS(d) = (COLS(d) | DIAGL(d) | DIAGR(d));

    DIAGL(d) <<= 1;
    DIAGR(d) >>= 1;

    while (d >= 0) {
#ifdef LOOKAHEAD
        int8_t l_rest = rest[L][d];
#endif
        uint_fast32_t posib = POSIBS(d); // backtrack ...

        while (posib != UINT_FAST32_MAX) {
            // The standard trick for getting the rightmost bit in the mask
            uint_fast32_t bit = ~posib & (posib + 1);
            posib ^= bit; // Eliminate the tried possibility.
            uint_fast32_t new_diagl = (bit << 1) | DIAGL(d);
            uint_fast32_t new_diagr = (bit >> 1) | DIAGR(d);
            bit |= COLS(d);
            uint_fast32_t new_posib = (bit | new_diagl | new_diagr);

            if (new_posib != UINT_FAST32_MAX) {
#ifdef LOOKAHEAD
                uint_fast32_t lookahead1 = (bit | (new_diagl << (LOOKAHEAD - 2)) | (new_diagr >> (LOOKAHEAD - 2)));
                uint_fast32_t lookahead2 = (bit | (new_diagl << (LOOKAHEAD - 1)) | (new_diagr >> (LOOKAHEAD - 1)));
                uint_fast32_t allowed2 = l_rest > (int8_t)0;

                if(allowed2 && ((lookahead2 == UINT_FAST32_MAX) || (lookahead1 == UINT_FAST32_MAX))) {
                    continue;
                }
#endif

              // The next two lines save stack depth + backtrack operations
              // when we passed the last possibility in a row.
              // Go lower in the stack, avoid branching by writing above the current
              // position
              POSIBS(d) = posib;
              d += posib != UINT_FAST32_MAX; // avoid branching with this trick
              posib = new_posib;
#ifdef LOOKAHEAD
              l_rest--;
#endif

              // make values current
              COLS(d) = bit;
              DIAGL(d) = new_diagl << 1;
              DIAGR(d) = new_diagr >> 1;
#ifdef LOOKAHEAD
              rest[L][d] = l_rest;
#endif
            } else {
                // when all columns are used, we found a solution
                num += bit == UINT_FAST32_MAX;
            }
        }
        d--;    // backtrack
    }

    out_cnt[G] += num;
}
