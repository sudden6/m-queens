typedef struct __attribute__ ((packed)) {
    uint cols; // bitfield with all the used columns
    uint diagl;// bitfield with all the used diagonals down left
    uint diagr;// bitfield with all the used diagonals down right
    //uint dummy;// dummy to align at 128bit
} start_condition;


/* Memory map for each stage

Stack Fill and stages start condition buffers
----------------------------------------------------------
|     Stack 0      |       ...        |     Stack N      |
----------------------------------------------------------
|   Stack Fill 0   |       ...        |   Stack Fill N   |
----------------------------------------------------------
| C0 |C1 | .. | CM | C0 |C1 | .. | CM | C0 |C1 | .. | CM |
----------------------------------------------------------

C0 .. CM -> start conditions stored in this buffer

In each stage there's a stack to store the results after this stage, for input into the
next stage. The number of Stacks must be a multiple of the local workgroup elements.
For each work-item in a workgroup there's a stack, that (currently) makes
Stack N = WORKGROUP_SIZE. WORKGROUP_SIZE is limited by the hardwares local memory, which
makes WORKGROUP_SIZE = 64 on AMD GCN hardware, and MAX_G_SIZE.

We launch one work-item per stack to fill the buffers as evenly as possible.

For each stack there's the Stack Fill value, which remembers how many elements are in the
stack. It must be able to count up to STACK_SIZE.

Each stack stores start Condition M = STACK_SIZE elements. This number is choosen, such
that at least WORKGROUP_SIZE + EXPANSION of that stage elements fit. This guarantees that
the next stage can pull WORKGROUP_SIZE elements and there's enough room for the previous
stages output in each stack.
*/

// maximum size of a local workgroup
#ifndef WORKGROUP_SIZE
#error "WORKGROUP_SIZE not defined"
#endif

// number of stacks in global memory
#ifndef N_STACKS
#error "N_STACKS not defined"
#endif

// number of elements in each stack
#ifndef STACK_SIZE
#error "STACK_SIZE not defined"
#endif

#define MAXN 29

typedef char int_fast8_t;
typedef uchar uint_fast8_t;
typedef short int_fast16_t;
typedef ushort uint_fast16_t;
typedef int int_fast32_t;
typedef uint uint_fast32_t;
#define UINT_FAST32_MAX UINT_MAX

typedef ulong uint_fast64_t;

#define DEBUG
//#define DEBUG printf


// maximum global work size, if exceeded synchronisation is broken
#define MAX_G_SIZE 64

// how many rows we go down in one step
#define DEPTH 2
// how many results are max in the next row
// x defines how many rows we went down from the first one
#define EXPANSION(x) (N - PLACED - 1 - x)

// max number of results we have after 2 steps down
#define MAX_RES (EXPANSION(0) * EXPANSION(1))

// N=27, PLACED=10
#define MAX_E MAX_RES

// for convenience
#define L (get_local_id(0))
#define G (get_global_id(0))

#define L_SIZE (get_local_size(0))
#define G_SIZE (get_global_size(0))

#define COL_MASK (~(UINT_FAST32_MAX << N))

// output array access

#define OUT_STACK_IDX(x) (G*STACK_SIZE + (x))

#define printf

kernel void first_step(__global const start_condition* in_starts, /* base of the input start conditions, G_SIZE*EXPANSION must not overflow output buffers */
                       __global start_condition* out_starts,      /* base of the output start conditions, must be N_STACKS * STACK_SIZE elements */
                       __global int* out_stack_idx		  /* base of the stack indizes, must be N_STACKS elements */
											 ) {

    __local uint_fast32_t cols[WORKGROUP_SIZE][DEPTH]; // Our backtracking 'stack'
    __local uint_fast32_t diagl[WORKGROUP_SIZE][DEPTH], diagr[WORKGROUP_SIZE][DEPTH];
    __local int_fast8_t rest[WORKGROUP_SIZE][DEPTH]; // number of rows left
    uint_fast32_t posibs = 0;
    int_fast8_t d = 0; // d is our depth in the backtrack stack
    uint l_out_stack_idx = out_stack_idx[G];
    // The UINT_FAST32_MAX here is used to fill all 'coloumn' bits after n ...
    cols[L][d] = in_starts[G].cols | (UINT_FAST32_MAX << N);
    if(popcount(cols[L][d]&COL_MASK) != PLACED) {
        printf("[F] wrong number of bits set: %d\n", popcount(cols[L][d]&COL_MASK));
    }
    DEBUG("F|IN  stage_idx: %d, cols: %x, set: %d\n", l_out_stack_idx, cols[L][d], popcount(cols[L][d]&COL_MASK));
    // This places the first two queens
    diagl[L][d] = in_starts[G].diagl;
    diagr[L][d] = in_starts[G].diagr;
#define LOOKAHEAD 3
/* TODO(sudden6): fix LOOKAHEAD implementation
#define REST_INIT (N - LOOKAHEAD - PLACED)
#define STOP_DEPTH (REST_INIT - DEPTH + 1) */

// TODO(sudden6): this is a backup STOP_DEPTH implementation
#define REST_INIT DEPTH
#define STOP_DEPTH 1

    // we're allready two rows into the field here
    rest[L][d] = REST_INIT;//in_starts[id].placed;

    //  The variable posib contains the bitmask of possibilities we still have
    //  to try in a given row ...
    uint_fast32_t posib = (cols[L][d] | diagl[L][d] | diagr[L][d]);

    while (d >= 0) {
      // moving the two shifts out of the inner loop slightly improves
      // performance
      uint_fast32_t diagl_shifted = diagl[L][d] << 1;
      uint_fast32_t diagr_shifted = diagr[L][d] >> 1;
      if(d >= DEPTH || d < 0) {
          printf("[F] d out of range #1\n");
      }
      int_fast8_t l_rest = rest[L][d];
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
            /*
            uint_fast32_t lookahead1 = (bit | (new_diagl << (LOOKAHEAD - 2)) | (new_diagr >> (LOOKAHEAD - 2)));
            uint_fast32_t lookahead2 = (bit | (new_diagl << (LOOKAHEAD - 1)) | (new_diagr >> (LOOKAHEAD - 1)));
            uint_fast8_t allowed1 = l_rest >= 0;
            uint_fast8_t allowed2 = l_rest > 0;


            if(allowed1 && (lookahead1 == UINT_FAST32_MAX)) {
                continue;
            }

            if(allowed2 && (lookahead2 == UINT_FAST32_MAX)) {
                continue;
            }*/

            if(l_rest == STOP_DEPTH) {
                out_starts[OUT_STACK_IDX(l_out_stack_idx)].cols = bit;
                //DEBUG("F|OUT gid: %d, cols: %x, set: %d d: %d posib: %x posibs: %x rest: %v2d\n",
                //             G, bit, popcount(bit&COL_MASK), d, posib, posibs, (rest[L][0], rest[L][1]));
                if(popcount(bit&COL_MASK) != 4) {
                    printf("[F] wrong number of bits set: %d\n", popcount(bit&COL_MASK));
                }
                out_starts[OUT_STACK_IDX(l_out_stack_idx)].diagl = new_diagl;
                out_starts[OUT_STACK_IDX(l_out_stack_idx)].diagr = new_diagr;
                l_out_stack_idx++;
                continue;
            }
						
            l_rest--;

            // The next two lines save stack depth + backtrack operations
            // when we passed the last possibility in a row.
            // Go lower in the stack, avoid branching by writing above the current
            // position
            posibs = posib;
            d += posib != UINT_FAST32_MAX; // avoid branching with this trick

            if(d >= DEPTH || d < 0) {
                printf("d out of range #2\n");
            }

            posib = new_posib;

            // make values current
            l_cols = bit;
            cols[L][d] = bit;
            diagl[L][d] = new_diagl;
            diagr[L][d] = new_diagr;
            rest[L][d] = l_rest;
            diagl_shifted = new_diagl << 1;
            diagr_shifted = new_diagr >> 1;
        } 
      }
      posib = posibs; // backtrack ...
      d--;
    }

    out_stack_idx[G] = l_out_stack_idx;
}


//*
#undef DEBUG
#define DEBUG
//*/

//#undef printf

kernel void inter_step(__global const start_condition* in_starts, /* base of the input start conditions, G_SIZE*EXPANSION must not overflow output buffers */
                       uint buffer_offset,                        /* input buffer number, must be 0 <= x < N_STACKS */
                       __global start_condition* out_starts,      /* base of the output start conditions, must be N_STACKS * STACK_SIZE elements */
                       __global int* out_stack_idx,            /* base of the stack indizes, must be N_STACKS elements */
                       __global int* in_stack_idx              /* base of the stack indizes, must be N_STACKS elements, will remove G_SIZE elements */
                      )
{

    __local uint_fast32_t cols[WORKGROUP_SIZE][DEPTH]; // Our backtracking 'stack'
    __local uint_fast32_t diagl[WORKGROUP_SIZE][DEPTH], diagr[WORKGROUP_SIZE][DEPTH];
    __local int_fast8_t rest[WORKGROUP_SIZE][DEPTH]; // number of rows left
    __local int old_in_fill;
    uint_fast32_t posibs;
    int_fast8_t d = 0; // d is our depth in the backtrack stack
    uint l_out_stack_idx = out_stack_idx[G];

    // handle stack fill update only in first work item
    if(L == 0) {
        // take elements from the input stack
        int take_items = min((int)(G_SIZE - G), (int) WORKGROUP_SIZE);
        old_in_fill = atomic_sub(&in_stack_idx[buffer_offset], take_items);
        //printf("G: %d, L_SIZE: %d\n", G, L_SIZE);
        int cur_stack_fill = old_in_fill - take_items;
        // check if we took to many
        if(cur_stack_fill < 0) {
            // took to many
            // fixup fill counter
            int res = atomic_xchg(&in_stack_idx[buffer_offset], 0);
            printf("G: %d, old_in_fill: %d, cur_stack_fill: %d, res: %d\n", G, old_in_fill, cur_stack_fill, res);
        }
    }
    // ensure all work-items have read the in_items value
    barrier(CLK_LOCAL_MEM_FENCE);

    int in_stack_item = old_in_fill - L - 1;
    // stop this work item when the stack has not enough items
    if(in_stack_item < 0) {
        return;
    }

    uint in_start_idx = buffer_offset * STACK_SIZE + in_stack_item;

    // The UINT_FAST32_MAX here is used to fill all 'coloumn' bits after n ...
    cols[L][d] = in_starts[in_start_idx].cols | (UINT_FAST32_MAX << N);

    int bitsset = popcount(cols[L][d]&COL_MASK);

    if(bitsset%2 != 0) {
        printf("[M] wrong number of bits set: %d\n", bitsset);
    }

    DEBUG("M|IN G: %d, lid: %d, cols: %x, set: %d\n", G, L, cols[L][d], bitsset);
    // This places the first two queens
    diagl[L][d] = in_starts[in_start_idx].diagl;
    diagr[L][d] = in_starts[in_start_idx].diagr;
#define LOOKAHEAD 3
/* TODO(sudden6): fix LOOKAHEAD implementation
#define REST_INIT (N - LOOKAHEAD - PLACED)
#define STOP_DEPTH (REST_INIT - DEPTH + 1)
*/

// TODO(sudden6): this is a backup STOP_DEPTH implementation
#define REST_INIT DEPTH
#define STOP_DEPTH 1

    // we're allready two rows into the field here
    rest[L][d] = REST_INIT;

    //  The variable posib contains the bitmask of possibilities we still have
    //  to try in a given row ...
    uint_fast32_t posib = (cols[L][d] | diagl[L][d] | diagr[L][d]);

    while (d >= 0) {
        if(d >= DEPTH || d < 0) {
            printf("d out of range #1\n");
        }
      // moving the two shifts out of the inner loop slightly improves
      // performance
      uint_fast32_t diagl_shifted = diagl[L][d] << 1;
      uint_fast32_t diagr_shifted = diagr[L][d] >> 1;
      int_fast8_t l_rest = rest[L][d];
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
        /* TODO(sudden6): fix LOOKAHEAD implementation
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
        */
            if(l_rest == STOP_DEPTH) {
                out_starts[OUT_STACK_IDX(l_out_stack_idx)].cols = bit;
                DEBUG("M|OUT G: %d, stack_idx: %d, set: %d, addr: %p\n",
                      G, l_out_stack_idx, popcount(bit&COL_MASK), &out_starts[OUT_STACK_IDX(l_out_stack_idx)]);
                if(popcount(bit&COL_MASK) != (bitsset + 2)) {
                    printf("[M] wrong number of bits set in output: %d, input: %d\n", popcount(bit&COL_MASK), bitsset);
                }
                out_starts[OUT_STACK_IDX(l_out_stack_idx)].diagl = new_diagl;
                out_starts[OUT_STACK_IDX(l_out_stack_idx)].diagr = new_diagr;
                l_out_stack_idx++;
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

            if(d >= DEPTH || d < 0) {
                printf("d out of range #2\n");
            }

            // make values current
            l_cols = bit;
            cols[L][d] = bit;
            diagl[L][d] = new_diagl;
            diagr[L][d] = new_diagr;
            rest[L][d] = l_rest;
            diagl_shifted = new_diagl << 1;
            diagr_shifted = new_diagr >> 1;
        }
      }
      posib = posibs; // backtrack ...
      d--;
    }

    out_stack_idx[G] = l_out_stack_idx;
}

#undef DEBUG
#define DEBUG

kernel void final_step(__global const start_condition* in_starts, /* input buffer base */
                       uint buffer_offset,                        /* input buffer number, must be 0 <= x < N_STACKS */
                       __global int* in_stack_idx,                /* base of the stack indizes, will remove G_SIZE elements */
                       __global uint* out_sum                     /* sum buffer base, must be G_SIZE elements */
                       ) {

    __local uint_fast32_t cols[WORKGROUP_SIZE][DEPTH]; // Our backtracking 'stack'
    __local uint_fast32_t diagl[WORKGROUP_SIZE][DEPTH], diagr[WORKGROUP_SIZE][DEPTH];
//    __local int_fast8_t rest[DEPTH]; // number of rows left
    __local int old_in_fill;
    uint_fast32_t num = 0;
    uint_fast32_t posibs;
    int_fast8_t d = 0; // d is our depth in the backtrack stack
    out_sum[G] = 0; // ensure unused output buffers are cleared

    // handle stack fill update only in first work item
    if(L == 0) {
        // take elements from the input stack
        int take_items = min((int)(G_SIZE - G), (int) WORKGROUP_SIZE);
        old_in_fill = atomic_sub(&in_stack_idx[buffer_offset], take_items);
        int cur_stack_fill = old_in_fill - take_items;
        // check if we took to many
        if(cur_stack_fill < 0) {
            // took to many
            // fixup fill counter
            int res = atomic_xchg(&in_stack_idx[buffer_offset], 0);
        }
    }
    // ensure all work-items have read the in_ites value
    barrier(CLK_LOCAL_MEM_FENCE);

    int in_stack_item = old_in_fill - L - 1;
    // stop this work item when the stack has not enough items
    if(in_stack_item < 0) {
        return;
    }

    uint in_start_idx = buffer_offset * STACK_SIZE + in_stack_item;

    // The UINT_FAST32_MAX here is used to fill all 'coloumn' bits after n ...
    cols[L][d] = in_starts[in_start_idx].cols | (UINT_FAST32_MAX << N);
    if((popcount(cols[L][d]&COL_MASK) != (N-1))
    && (popcount(cols[L][d]&COL_MASK) != (N-2))) {
        //printf("[L] wrong number of bits set: %d\n", popcount(cols[L][d]&COL_MASK));
    }
    DEBUG("L|IN  lid: %d, gid: %d, cols: %x, old_fill: %u, buf_off: %u, in_idx: %u, addr: %p\n",
           L, G, cols[L][d], old_in_fill, buffer_offset, in_start_idx, &in_starts[in_start_idx]);
    // This places the first two queens
    diagl[L][d] = in_starts[in_start_idx].diagl;
    diagr[L][d] = in_starts[in_start_idx].diagr;
#define LOOKAHEAD 3
/* TODO(sudden6): check if lookahead even works for last 2 steps
#define REST_INIT (N - LOOKAHEAD - PLACED)
#define STOP_DEPTH (REST_INIT - DEPTH + 1)

    // we're allready two rows into the field here
    rest[d] = REST_INIT;//in_starts[id].placed; */

    //  The variable posib contains the bitmask of possibilities we still have
    //  to try in a given row ...
    uint_fast32_t posib = (cols[L][d] | diagl[L][d] | diagr[L][d]);

    while (d >= 0) {

        if(d >= DEPTH || d < 0) {
            //printf("[L] d out of range #1\n");
        }
      // moving the two shifts out of the inner loop slightly improves
      // performance
      uint_fast32_t diagl_shifted = diagl[L][d] << 1;
      uint_fast32_t diagr_shifted = diagr[L][d] >> 1;
      //int_fast8_t l_rest = rest[d];
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
            /*
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

            l_rest--;
            */
            // The next two lines save stack depth + backtrack operations
            // when we passed the last possibility in a row.
            // Go lower in the stack, avoid branching by writing above the current
            // position
            posibs = posib;
            d += posib != UINT_FAST32_MAX; // avoid branching with this trick
            posib = new_posib;

            if(d >= DEPTH || d < 0) {
                //printf("[L] d out of range #2\n");
            }

            // make values current
            l_cols = bit;
            cols[L][d] = bit;
            diagl[L][d] = new_diagl;
            diagr[L][d] = new_diagr;
            //rest[d] = l_rest;
            diagl_shifted = new_diagl << 1;
            diagr_shifted = new_diagr >> 1;
        } else {
#undef DEBUG
#define DEBUG
            // when all columns are used, we found a solution
            num += bit == UINT_FAST32_MAX;
            if(bit == UINT_FAST32_MAX) {
                DEBUG("L|OUT lid: %d, gid: %d, cols: %x\n", L, G, bit);
            }
        }
      }
      posib = posibs; // backtrack ...
      d--;
    }
    out_sum[G] += num;
}
