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

// number of elements in each input stack
#ifndef IN_STACK_SIZE
#error "IN_STACK_SIZE not defined"
#endif

// number of elements in each output stack
#ifndef OUT_STACK_SIZE
#error "OUT_STACK_SIZE not defined"
#endif

// how many rows we go down in one step
#ifndef DEPTH
#error "DEPTH not defined"
#endif

typedef char int_fast8_t;
typedef uchar uint_fast8_t;
typedef short int_fast16_t;
typedef ushort uint_fast16_t;
typedef int int_fast32_t;
typedef uint uint_fast32_t;
#define UINT_FAST32_MAX UINT_MAX

typedef ulong uint_fast64_t;

// maximum global work size, if exceeded synchronisation is broken
#define MAX_G_SIZE 64

// for convenience
#define L (get_local_id(0))
#define G (get_global_id(0))

#define L_SIZE (get_local_size(0))
#define G_SIZE (get_global_size(0))

#define COL_MASK (~(UINT_FAST32_MAX << N))

// output array access

#define OUT_STACK_IDX(x) (G*STACK_SIZE + (x))

#define ASSERT
//#define DEBUG

kernel void first_step(__global const start_condition* in_starts, /* base of the input start conditions, G_SIZE*EXPANSION must not overflow output buffers */
                       __global start_condition* out_starts,      /* base of the output start conditions, must be N_STACKS * STACK_SIZE elements */
                       __global ushort* out_stack_idx                /* base of the stack indizes, must be N_STACKS elements */
											 ) {

    __local uint_fast32_t cols[WORKGROUP_SIZE][DEPTH]; // Our backtracking 'stack'
    __local uint_fast32_t diagl[WORKGROUP_SIZE][DEPTH];
    __local uint_fast32_t diagr[WORKGROUP_SIZE][DEPTH];
    __local uint_fast32_t posibs[WORKGROUP_SIZE][DEPTH];

    //uint_fast32_t posibs = 0;
    int_fast8_t d = 0; // d is our depth in the backtrack stack
    uint l_out_stack_idx = G*OUT_STACK_SIZE + out_stack_idx[G];
    // The UINT_FAST32_MAX here is used to fill all 'coloumn' bits after n ...
    cols[L][d] = in_starts[G].cols | (UINT_FAST32_MAX << N);
#ifdef ASSERT
    if(popcount(cols[L][d]&COL_MASK) != PLACED) {
        printf("[F] entry: wrong number of bits set: %d\n", popcount(cols[L][d]&COL_MASK));
    }
#endif
#ifdef DEBUG
    printf("F|IN  gid: %d, stage_idx: %d, cols: %x, set: %d\n", G, l_out_stack_idx, cols[L][d], popcount(cols[L][d]&COL_MASK));
#endif

    // This places the first two queens
    diagl[L][d] = in_starts[G].diagl;
    diagr[L][d] = in_starts[G].diagr;
//#define LOOKAHEAD 3
#define STOP_DEPTH (PLACED + DEPTH)

    //  The variable posib contains the bitmask of possibilities we still have
    //  to try in a given row ...
    uint_fast32_t posib = (cols[L][d] | diagl[L][d] | diagr[L][d]);

    while (d >= 0) {
      // moving the two shifts out of the inner loop slightly improves
      // performance
      uint_fast32_t diagl_shifted = diagl[L][d] << 1;
      uint_fast32_t diagr_shifted = diagr[L][d] >> 1;
#ifdef ASSERT
      if(d >= DEPTH || d < 0) {
          printf("[F] d out of range #1\n");
      }
#endif

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
#ifdef LOOKAHEAD
#if (PLACED + DEPTH) < (N-1)
            uint_fast32_t lookahead1 = (bit | (new_diagl << (LOOKAHEAD - 2)) | (new_diagr >> (LOOKAHEAD - 2)));

            if(lookahead1 == UINT_FAST32_MAX) {
                continue;
            }
#endif

#if (PLACED + DEPTH) < (N-2)
            uint_fast32_t lookahead2 = (bit | (new_diagl << (LOOKAHEAD - 1)) | (new_diagr >> (LOOKAHEAD - 1)));
            if(lookahead2 == UINT_FAST32_MAX) {
                continue;
            }
#endif
#endif

            // high bits are set to 1, add this number
            if(popcount(bit) == (32 - N + STOP_DEPTH)) {
                out_starts[l_out_stack_idx].cols = bit;
#ifdef DEBUG
                printf("F|OUT gid: %d, cols: %x, set: %d, out_stack_idx: %d\n",
                             G, bit, popcount(bit&COL_MASK), l_out_stack_idx);
#endif
#ifdef ASSERT
                if(popcount(bit&COL_MASK) != (PLACED + DEPTH)) {
                    printf("[F] exit: wrong number of bits set: %d\n", popcount(bit&COL_MASK));
                }
#endif
                out_starts[l_out_stack_idx].diagl = new_diagl;
                out_starts[l_out_stack_idx].diagr = new_diagr;
                l_out_stack_idx++;
                continue;
            }

            // The next two lines save stack depth + backtrack operations
            // when we passed the last possibility in a row.
            // Go lower in the stack, avoid branching by writing above the current
            // position
            posibs[L][d + 1] = posib;
            d += posib != UINT_FAST32_MAX; // avoid branching with this trick

#ifdef ASSERT
            if(d >= DEPTH || d < 0) {
                printf("d out of range #2\n");
            }
#endif

            posib = new_posib;

            // make values current
            l_cols = bit;
            cols[L][d] = bit;
            diagl[L][d] = new_diagl;
            diagr[L][d] = new_diagr;

            diagl_shifted = new_diagl << 1;
            diagr_shifted = new_diagr >> 1;
        } 
      }
      posib = posibs[L][d]; // backtrack ...
      d--;
    }

    out_stack_idx[G] = l_out_stack_idx - G*OUT_STACK_SIZE;
#ifdef DEBUG
    printf("F|FIN gid: %d, out_stack_idx: %d\n", G, out_stack_idx[G]);
#endif
}



#define FULL_RUN
//#define DEBUG
#if 0
kernel void inter_step(__global start_condition* in_starts,     /* base of the input start conditions, will work on elements at IN_STACK_SIZE*G */
                       uint max_runs,                           /* maximum amount of elements to take from input buffer */
                       __global start_condition* out_starts,    /* base of the output start conditions */
                       __global ushort* out_stack_idx,             /* base of the stack indizes, must be N_STACKS elements */
                       __global ushort* in_stack_idx               /* base of the stack indizes, must be N_STACKS elements, will remove at most max_runs elements */
                      )
{


    __local uint_fast32_t cols[WORKGROUP_SIZE][DEPTH]; // Our backtracking 'stack'
    __local uint_fast32_t diagl[WORKGROUP_SIZE][DEPTH];
    __local uint_fast32_t diagr[WORKGROUP_SIZE][DEPTH];
    __local uint_fast32_t posibs[WORKGROUP_SIZE][DEPTH];

    uint l_out_stack_idx = OUT_STACK_SIZE*G + out_stack_idx[G];
    uint l_in_stack_idx = IN_STACK_SIZE*G + in_stack_idx[G];

    if(in_stack_idx[G] == 0) {
        return;
    }

    for(size_t run = 0; run < max_runs; run++) {

        l_in_stack_idx--;

        int_fast8_t d = 0; // d is our depth in the backtrack stack
        // The UINT_FAST32_MAX here is used to fill all 'coloumn' bits after n ...
        cols[L][d] = in_starts[l_in_stack_idx].cols | (UINT_FAST32_MAX << N);

        // This places the first two queens
        diagl[L][d] = in_starts[l_in_stack_idx].diagl;
        diagr[L][d] = in_starts[l_in_stack_idx].diagr;
        in_starts[l_in_stack_idx].cols = 0;
        in_starts[l_in_stack_idx].diagl = 0;
        in_starts[l_in_stack_idx].diagr = 0;

#ifdef ASSERT
        int bitsset = popcount(cols[L][d]&COL_MASK);
        if(bitsset != PLACED) {
            printf("[M] entry: wrong number of bits set: %d, STAGE: %d, G: %d\n", bitsset, STAGE_IDX, G);
        }
#endif
#ifdef DEBUG
        printf("M|IN G: %d, lid: %d, cols: %x, max_runs: %d, in_stack_idx: %d\n", G, L, cols[L][d], max_runs, l_in_stack_idx);
#endif
#undef LOOKAHEAD

#define LOOKAHEAD 3
#define STOP_DEPTH (PLACED + DEPTH)

        uint_fast32_t posib = (cols[L][d] | diagl[L][d] | diagr[L][d]);

        while (d >= 0) {
#ifdef ASSERT
            if(d >= DEPTH || d < 0) {
                printf("d out of range #1\n");
            }
#endif
          // moving the two shifts out of the inner loop slightly improves
          // performance
          uint_fast32_t diagl_shifted = diagl[L][d] << 1;
          uint_fast32_t diagr_shifted = diagr[L][d] >> 1;

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
#ifdef LOOKAHEAD
#if (PLACED + DEPTH) < (N-1)
                uint_fast32_t lookahead1 = (bit | (new_diagl << (LOOKAHEAD - 2)) | (new_diagr >> (LOOKAHEAD - 2)));
                if(lookahead1 == UINT_FAST32_MAX) {
                    continue;
                }
#endif

#if (PLACED+DEPTH) < (N-2)
                uint_fast32_t lookahead2 = (bit | (new_diagl << (LOOKAHEAD - 1)) | (new_diagr >> (LOOKAHEAD - 1)));
                if(lookahead2 == UINT_FAST32_MAX) {
                    continue;
                }
#endif

#endif
                if(popcount(bit) == (32 - N + STOP_DEPTH)) {
#ifdef DEBUG
                    printf("M|OUT G: %d, stack_idx: %d, set: %d\n",
                          G, l_out_stack_idx, popcount(bit&COL_MASK));
#endif
#ifdef ASSERT
                    if(popcount(bit&COL_MASK) != (bitsset + DEPTH)) {
                        printf("[M] exit: wrong number of bits set in output: %d, input: %d\n", popcount(bit&COL_MASK), bitsset);
                    }
#endif
                    out_starts[l_out_stack_idx].cols = bit;
                    out_starts[l_out_stack_idx].diagl = new_diagl;
                    out_starts[l_out_stack_idx].diagr = new_diagr;
                    l_out_stack_idx++;
                    continue;
                }

                // The next two lines save stack depth + backtrack operations
                // when we passed the last possibility in a row.
                // Go lower in the stack, avoid branching by writing above the current
                // position
                posibs[L][d + 1] = posib;
                d += posib != UINT_FAST32_MAX; // avoid branching with this trick
                posib = new_posib;

#ifdef ASSERT
                if(d >= DEPTH || d < 0) {
                    printf("d out of range #2\n");
                }
#endif

                // make values current
                l_cols = bit;
                cols[L][d] = bit;
                diagl[L][d] = new_diagl;
                diagr[L][d] = new_diagr;
                diagl_shifted = new_diagl << 1;
                diagr_shifted = new_diagr >> 1;
            }
          }
          posib = posibs[L][d]; // backtrack ...
          d--;
        }
        if(l_in_stack_idx == G*IN_STACK_SIZE) {
            break;
        }
    }

    in_stack_idx[G] = l_in_stack_idx - G*IN_STACK_SIZE;
    out_stack_idx[G] = l_out_stack_idx - G*OUT_STACK_SIZE;
}

#endif

kernel void final_step(__global start_condition* in_starts, /* input buffer base */
                       __global ushort* in_stack_idx,                /* base of the stack indizes, will remove G_SIZE elements */
                       __global uint* out_sum                     /* sum buffer base, must be G_SIZE elements */
                       ) {

#undef LOOKAHEAD

    __local uint_fast32_t cols[WORKGROUP_SIZE][DEPTH]; // Our backtracking 'stack'
    __local uint_fast32_t diagl[WORKGROUP_SIZE][DEPTH];
    __local uint_fast32_t diagr[WORKGROUP_SIZE][DEPTH];
    __local uint_fast32_t posibs[WORKGROUP_SIZE][DEPTH];

    uint_fast32_t num = 0;

    if(in_stack_idx[G] == 0) {
        return;
    }

    uint l_in_stack_idx = IN_STACK_SIZE*G + in_stack_idx[G];


    while(l_in_stack_idx > G*IN_STACK_SIZE)
    {
        l_in_stack_idx--;

        int_fast8_t d = 0; // d is our depth in the backtrack stack

        // The UINT_FAST32_MAX here is used to fill all 'coloumn' bits after n ...
        cols[L][d] = in_starts[l_in_stack_idx].cols | (UINT_FAST32_MAX << N);

        // This places the first two queens
        diagl[L][d] = in_starts[l_in_stack_idx].diagl;
        diagr[L][d] = in_starts[l_in_stack_idx].diagr;

        in_starts[l_in_stack_idx].cols = 0;
        in_starts[l_in_stack_idx].diagl = 0;
        in_starts[l_in_stack_idx].diagr = 0;

#ifdef ASSERT
        uint bitsset = popcount(cols[L][d]&COL_MASK);
        if(bitsset != PLACED) {
            printf("[L] entry: wrong number of bits set: %d\n", bitsset);
        }
#endif
#ifdef DEBUG
        printf("L|IN  lid: %d, gid: %d, cols: %x, set: %d, in_idx: %u\n",
               L, G, cols[L][d], popcount(cols[L][d]&COL_MASK), l_in_stack_idx);
#endif

        //  The variable posib contains the bitmask of possibilities we still have
        //  to try in a given row ...
        uint_fast32_t posib = (cols[L][d] | diagl[L][d] | diagr[L][d]);

        while (d >= 0) {
#ifdef ASSERT
            if(d >= DEPTH || d < 0) {
                printf("[L] d out of range #1\n");
            }
#endif
          // moving the two shifts out of the inner loop slightly improves
          // performance
          uint_fast32_t diagl_shifted = diagl[L][d] << 1;
          uint_fast32_t diagr_shifted = diagr[L][d] >> 1;
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
                // The next two lines save stack depth + backtrack operations
                // when we passed the last possibility in a row.
                // Go lower in the stack, avoid branching by writing above the current
                // position
                posibs[L][d + 1] = posib;
                d += posib != UINT_FAST32_MAX; // avoid branching with this trick
                posib = new_posib;
#ifdef ASSERT
                if(d >= DEPTH || d < 0) {
                    printf("[L] d out of range #2\n");
                }
#endif

                // make values current
                l_cols = bit;
                cols[L][d] = bit;
                diagl[L][d] = new_diagl;
                diagr[L][d] = new_diagr;

                diagl_shifted = new_diagl << 1;
                diagr_shifted = new_diagr >> 1;
            } else {
                // when all columns are used, we found a solution
                num += bit == UINT_FAST32_MAX;
#ifdef DEBUG
                if(bit == UINT_FAST32_MAX) {
                    DEBUG("L|OUT lid: %d, gid: %d, cols: %x\n", L, G, bit);
                }
#endif
            }
          }
          posib = posibs[L][d]; // backtrack ...
          d--;
        }
    }
    in_stack_idx[G] = l_in_stack_idx - G*IN_STACK_SIZE;
    out_sum[G] += num;
#ifdef DEBUG
    printf("L|FIN gid: %d, num: %d\n", G, num);
#endif
}
