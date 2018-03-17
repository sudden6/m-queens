typedef struct __attribute__ ((packed)) {
    uint cols; // bitfield with all the used columns
    uint diagl;// bitfield with all the used diagonals down left
    uint diagr;// bitfield with all the used diagonals down right
    //uint dummy;// dummy to align at 128bit
} start_condition;

#define MAXN 29

typedef char int_fast8_t;
typedef uchar uint_fast8_t;
typedef short int_fast16_t;
typedef ushort uint_fast16_t;
typedef int int_fast32_t;
typedef uint uint_fast32_t;
#define UINT_FAST32_MAX UINT_MAX

typedef ulong uint_fast64_t;

#define WORKGROUP_SIZE 64

// how many rows we go down in one step
#define DEPTH 2
// how many results are max in the next row
// x defines how many rows we went down from the first one
#define EXPANSION(x) (N - PLACED - 1 - x)

// max number of results we have after 2 steps down
#define MAX_RES (EXPANSION(0) * EXPANSION(1))

// N=27, PLACED=10
#define MAX_E MAX_RES

#define MAXD (N - PLACED)

#if (MAXD > 13)
#error "Depth to high, risk of overflow in result counter"
#endif

// for convenience
#define L (get_local_id(0))
#define G (get_global_id(0))

// res_cnt must be 1 element
kernel void single_step(__global const start_condition* in_starts,
                        __global start_condition* out_starts,
                        __global uint* out_res,
                        __local uint* res_cnts) {

    __local uint_fast32_t cols[WORKGROUP_SIZE][DEPTH]; // Our backtracking 'stack'
    __local uint_fast32_t diagl[WORKGROUP_SIZE][DEPTH], diagr[WORKGROUP_SIZE][DEPTH];
    __local int_fast8_t rest[DEPTH]; // number of rows left
		uint_fast32_t posibs;
    int_fast8_t d = 0; // d is our depth in the backtrack stack
    // The UINT_FAST32_MAX here is used to fill all 'coloumn' bits after n ...
    cols[L][d] = in_starts[G].cols | (UINT_FAST32_MAX << N);
    // This places the first two queens
    diagl[L][d] = in_starts[G].diagl;
    diagr[L][d] = in_starts[G].diagr;
#define LOOKAHEAD 3
#define REST_INIT (N - LOOKAHEAD - PLACED)
#define STOP_DEPTH (REST_INIT - DEPTH + 1)

    // we're allready two rows into the field here
    rest[d] = REST_INIT;//in_starts[id].placed;

    //  The variable posib contains the bitmask of possibilities we still have
    //  to try in a given row ...
    uint_fast32_t posib = (cols[L][d] | diagl[L][d] | diagr[L][d]);

    while (d >= 0) {
      // moving the two shifts out of the inner loop slightly improves
      // performance
      uint_fast32_t diagl_shifted = diagl[L][d] << 1;
      uint_fast32_t diagr_shifted = diagr[L][d] >> 1;
      int_fast8_t l_rest = rest[d];
      uint_fast32_t l_cols = cols[L][d];

      while (posib != UINT_FAST32_MAX) {
        // The standard trick for getting the rightmost bit in the mask
        uint_fast32_t bit = ~posib & (posib + 1);
        uint_fast32_t new_diagl = (bit << 1) | diagl_shifted;
        uint_fast32_t new_diagr = (bit >> 1) | diagr_shifted;
        uint_fast32_t new_posib = (l_cols | bit | new_diagl | new_diagr);
        posib ^= bit; // Eliminate the tried possibility.
        bit |= l_cols;

        if (new_posib != UINT_FAST32_MAX) {
            uint_fast32_t lookahead1 = (bit | (new_diagl << (LOOKAHEAD - 2)) | (new_diagr >> (LOOKAHEAD - 2)));
            uint_fast32_t lookahead2 = (bit | (new_diagl << (LOOKAHEAD - 1)) | (new_diagr >> (LOOKAHEAD - 1)));
            uint_fast8_t allowed1 = l_rest >= 0;
            uint_fast8_t allowed2 = l_rest > 0;


            if(allowed1 && (lookahead1 == UINT_FAST32_MAX)) {
                continue;
            }

            if(allowed2 && (lookahead2 == UINT_FAST32_MAX)) {
                continue;
            }
						
            if(l_rest == STOP_DEPTH) {
                uint l_idx = atomic_add(res_cnts, 1);
                uint g_idx = G * MAX_E * WORKGROUP_SIZE + l_idx;
                out_starts[g_idx].cols = bit;
                out_starts[g_idx].diagl = new_diagl;
                out_starts[g_idx].diagr = new_diagr;
                continue;
            }
						
            l_rest--;

            // The next two lines save stack depth + backtrack operations
            // when we passed the last possibility in a row.
            // Go lower in the stack, avoid branching by writing above the current
            // position
            posibs = posib;
            d += posib != UINT_FAST32_MAX; // avoid branching with this trick
            posib = new_posib;

            // make values current
            l_cols = bit;
            cols[L][d] = bit;
            diagl[L][d] = new_diagl;
            diagr[L][d] = new_diagr;
            rest[d] = l_rest;
            diagl_shifted = new_diagl << 1;
            diagr_shifted = new_diagr >> 1;
        } 
      }
      posib = posibs; // backtrack ...
      d--;
    }
}
