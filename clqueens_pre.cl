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

//#define ASSERT

// pack two nibbles into one byte
uint get_nibble(__local unsigned char* byte, uint index) {
    return (*byte >> (index*4) & 0xF);
}

void set_nibble(__local unsigned char* byte, uint index, unsigned char value) {
    unsigned char tmp = (value & 0xF) << index*4;
    unsigned char mask = 0xF << index * 4;
    *byte &= ~mask;
    *byte |= tmp;
}

// implement a one byte counter with three storage locations
// split an uint into three parts:
// byte2 | byte1 | byte0 | cnt |
// store cnt to byte0 or byte1
void store_cnt(uint* var, unsigned char pos) {
    //printf("[%03d][B] store| var: %08x, pos: %d\n", G, *var, pos);
    uint tmp = bitselect((uint) 0, (uint) *var, (uint) 0xFF);
	tmp = tmp << ((pos + 1) * 8);
	*var = bitselect((uint) *var, (uint) tmp, (uint) 0xFF << ((pos + 1)*8));
    //printf("[%03d][A] store| var: %08x, pos: %d\n", G, *var, pos);
}

// load cnt from byte0 or byte1
void load_cnt(uint* var, unsigned char pos) {
    //printf("[%03d][B] load | var: %08x, pos: %d\n", G, *var, pos);
	uint tmp = bitselect((uint) 0, (uint) *var, (uint) 0xFF << ((pos + 1)*8));
	tmp = tmp >> ((pos + 1) * 8);
	*var = bitselect((uint) *var, (uint) tmp, (uint) 0xFF);
    //printf("[%03d][A] load | var: %08x, pos: %d\n", G, *var, pos);
}

// subtract one from cnt with correct overflow and underflow handling
void sub_cnt(uint* var) {
    //printf("[%03d][B] sub  | var: %08x\n", G, *var);
    uint tmp = (*var & 0xFF);
    tmp--;
    *var = bitselect((uint) *var, (uint) tmp, (uint) 0xFF);
    //printf("[%03d][A] sub  | var: %08x\n", G, *var);
}

//#define DEBUG

kernel void first_step(__global const start_condition* in_starts, /* base of the input start conditions, G_SIZE*EXPANSION must not overflow output buffers */
                       __global start_condition* out_starts,      /* base of the output start conditions, must be N_STACKS * STACK_SIZE elements */
                       __global int* out_stack_idx		  /* base of the stack indizes, must be N_STACKS elements */
											 ) {

    __local uint_fast32_t cols[WORKGROUP_SIZE][DEPTH]; // Our backtracking 'stack'
    __local uint_fast32_t diagl[WORKGROUP_SIZE][DEPTH], diagr[WORKGROUP_SIZE][DEPTH];
    //__local int_fast8_t rest[WORKGROUP_SIZE]; // number of rows left
    //uint rest = 0;
    uint_fast32_t posibs = 0;
    int_fast8_t d = 0; // d is our depth in the backtrack stack
    uint l_out_stack_idx = G*STACK_SIZE + out_stack_idx[G];
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
#ifdef LOOKAHEAD
//#define REST_INIT (N - LOOKAHEAD - PLACED)
//#define STOP_DEPTH (REST_INIT - DEPTH + 1)
#else
// this is a backup STOP_DEPTH implementation
#define REST_INIT DEPTH
#endif

#define STOP_DEPTH (PLACED + DEPTH)


    // we're allready two rows into the field here
    //set_nibble(&rest[L], d, REST_INIT);
    //rest = REST_INIT;
    //store_cnt(&rest, d);
    //uchar2 rest_store = 0;
    //rest_store.s0 = REST_INIT;

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
      //int_fast8_t l_rest = get_nibble(&rest[L], d);
      //load_cnt(&rest, d);
      //uchar l_rest = d ? rest_store.s1 : rest_store.s0;

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
            uint_fast32_t lookahead1 = (bit | (new_diagl << (LOOKAHEAD - 2)) | (new_diagr >> (LOOKAHEAD - 2)));
            //uint_fast8_t allowed1 =  >= 0;
            //uint_fast8_t allowed2 = popcount(bit & COL_MASK) < (N-2);


            if(lookahead1 == UINT_FAST32_MAX) {
                continue;
            }

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
                printf("F|OUT gid: %d, cols: %x, set: %d d: %d posib: %x posibs: %x\n",
                             G, bit, popcount(bit&COL_MASK), d, posib, posibs);
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
						
            //l_rest--;
            //sub_cnt(&rest);
#ifdef DEBUG_1
            uint tmp = rest & 0xFF;
            if(tmp != 0 && tmp != 1 && tmp != 2) {
                printf("[%03d] O_o tmp: %x\n", G, tmp);
            }
#endif

            // The next two lines save stack depth + backtrack operations
            // when we passed the last possibility in a row.
            // Go lower in the stack, avoid branching by writing above the current
            // position
            posibs = posib;
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
            //set_nibble(&rest[L], d, l_rest);
            //store_cnt(&rest, d);
            /*if(d) {
                rest_store.s1 = l_rest;
            } else {
                rest_store.s0 = l_rest;
            }*/
            diagl_shifted = new_diagl << 1;
            diagr_shifted = new_diagr >> 1;
        } 
      }
      posib = posibs; // backtrack ...
      d--;
    }

#ifdef DEBUG
    printf("F|FIN gid: %d\n");
#endif
    out_stack_idx[G] = l_out_stack_idx - G*STACK_SIZE;
}



#define FULL_RUN
//#define DEBUG

kernel void inter_step(__global start_condition* in_starts, /* base of the input start conditions, G_SIZE*EXPANSION must not overflow output buffers */
                       uint buffer_offset,                        /* input buffer number, must be 0 <= x < N_STACKS */
                       __global start_condition* out_starts,      /* base of the output start conditions, must be N_STACKS * STACK_SIZE elements */
                       __global int* out_stack_idx,            /* base of the stack indizes, must be N_STACKS elements */
                       __global int* in_stack_idx              /* base of the stack indizes, must be N_STACKS elements, will remove G_SIZE elements */
                      )
{

    __local uint_fast32_t cols[WORKGROUP_SIZE][DEPTH]; // Our backtracking 'stack'
    __local uint_fast32_t diagl[WORKGROUP_SIZE][DEPTH], diagr[WORKGROUP_SIZE][DEPTH];
    //__local int_fast8_t rest[WORKGROUP_SIZE][DEPTH]; // number of rows left
    uint_fast32_t posibs;
    int_fast8_t d = 0; // d is our depth in the backtrack stack
    uint l_out_stack_idx = out_stack_idx[G];

#ifndef FULL_RUN
    __local int old_in_fill;
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
#elsif 0
    // stack index was already decreased by host, add own Gid here to access
    int stack_element = in_stack_idx[buffer_offset] + G;
#ifdef ASSERT
    if (stack_element < 0) {
        printf("Stack underrun, tried to access idx: %d\n", stack_element);
    }
#endif
    uint in_start_idx = buffer_offset * STACK_SIZE + stack_element;
#endif
    uint in_start_idx = buffer_offset + G;

    // The UINT_FAST32_MAX here is used to fill all 'coloumn' bits after n ...
    cols[L][d] = in_starts[in_start_idx].cols | (UINT_FAST32_MAX << N);

    // This places the first two queens
    diagl[L][d] = in_starts[in_start_idx].diagl;
    diagr[L][d] = in_starts[in_start_idx].diagr;
    in_starts[in_start_idx].cols = 0;
    in_starts[in_start_idx].diagl = 0;
    in_starts[in_start_idx].diagr = 0;


#ifdef ASSERT
    int bitsset = popcount(cols[L][d]&COL_MASK);
    if(bitsset != PLACED) {
        printf("[M] entry: wrong number of bits set: %d, STAGE: %d, G: %d\n", bitsset, STAGE_IDX, G);
    }
#endif
#ifdef DEBUG
    printf("M|IN G: %d, lid: %d, cols: %x, buf_idx: %d\n", G, L, cols[L][d], buffer_offset);
#endif
#undef LOOKAHEAD
#define LOOKAHEAD 3
#ifdef LOOKAHEAD
//#define REST_INIT (N - LOOKAHEAD - PLACED)
//#define STOP_DEPTH (REST_INIT - DEPTH + 1)
#else
// this is a backup STOP_DEPTH implementation
#define REST_INIT DEPTH
#endif

#define STOP_DEPTH (PLACED + DEPTH)


    // we're allready two rows into the field here
    //rest[L][d] = REST_INIT;
    //uint rest = REST_INIT;
    //store_cnt(&rest, d);
    //uchar2 rest_store = 0;
    //rest_store.s0 = REST_INIT;

    //  The variable posib contains the bitmask of possibilities we still have
    //  to try in a given row ...
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
      //int_fast8_t l_rest = rest[L][d];
      //uchar l_rest = d ? rest_store.s1 : rest_store.s0;
      //load_cnt(&rest, d);
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
            uint_fast32_t lookahead1 = (bit | (new_diagl << (LOOKAHEAD - 2)) | (new_diagr >> (LOOKAHEAD - 2)));
            //uint_fast32_t lookahead2 = (bit | (new_diagl << (LOOKAHEAD - 1)) | (new_diagr >> (LOOKAHEAD - 1)));
            //uint_fast8_t allowed1 = l_rest >= 0;
            //uint_fast8_t allowed2 = l_rest > 0;

            //uint_fast8_t allowed1 = (rest & 0xFF) >= 0;
            //uint_fast8_t allowed2 = (rest & 0xFF) > 0;
            if(lookahead1 == UINT_FAST32_MAX) {
                continue;
            }

#if (PLACED+DEPTH) < (N-2)
            uint_fast32_t lookahead2 = (bit | (new_diagl << (LOOKAHEAD - 1)) | (new_diagr >> (LOOKAHEAD - 1)));
            if(lookahead2 == UINT_FAST32_MAX) {
                continue;
            }
#endif



#endif
            if(popcount(bit) == (32 - N + STOP_DEPTH)) {
                out_starts[OUT_STACK_IDX(l_out_stack_idx)].cols = bit;
#ifdef DEBUG
                printf("M|OUT G: %d, stack_idx: %d, set: %d, addr: %p\n",
                      G, l_out_stack_idx, popcount(bit&COL_MASK), &out_starts[OUT_STACK_IDX(l_out_stack_idx)]);
#endif
#ifdef ASSERT
                if(popcount(bit&COL_MASK) != (bitsset + DEPTH)) {
                    printf("[M] exit: wrong number of bits set in output: %d, input: %d\n", popcount(bit&COL_MASK), bitsset);
                }
#endif
                out_starts[OUT_STACK_IDX(l_out_stack_idx)].diagl = new_diagl;
                out_starts[OUT_STACK_IDX(l_out_stack_idx)].diagr = new_diagr;
                l_out_stack_idx++;
                continue;
            }

            //l_rest--;

            // The next two lines save stack depth + backtrack operations
            // when we passed the last possibility in a row.
            // Go lower in the stack, avoid branching by writing above the current
            // position
            posibs = posib;
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
            //rest[L][d] = l_rest;
            //store_cnt(&rest, d);
            /*
            if(d) {
                rest_store.s1 = l_rest;
            } else {
                rest_store.s0 = l_rest;
            }*/
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

kernel void final_step(__global start_condition* in_starts, /* input buffer base */
                       uint buffer_offset,                        /* input buffer number, must be 0 <= x < N_STACKS */
                       __global int* in_stack_idx,                /* base of the stack indizes, will remove G_SIZE elements */
                       __global uint* out_sum                     /* sum buffer base, must be G_SIZE elements */
                       ) {

#undef LOOKAHEAD
//#define LOOKAHEAD 3

    __local uint_fast32_t cols[WORKGROUP_SIZE][DEPTH]; // Our backtracking 'stack'
    __local uint_fast32_t diagl[WORKGROUP_SIZE][DEPTH], diagr[WORKGROUP_SIZE][DEPTH];
#ifdef LOOKAHEAD
    __local int_fast8_t rest[WORKGROUP_SIZE][DEPTH]; // number of rows left
#endif
    uint_fast32_t num = 0;
    uint_fast32_t posibs;
    int_fast8_t d = 0; // d is our depth in the backtrack stack

#ifndef FULL_RUN
    __local int old_in_fill;
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
#elsif 0
    // stack index was already decreased by host, add own Gid here for access
    int stack_element = in_stack_idx[buffer_offset] + G;
#ifdef ASSERT
    if (stack_element < 0) {
        printf("Stack underrun, tried to access idx: %d\n", stack_element);
    }
#endif
    uint in_start_idx = buffer_offset * STACK_SIZE + stack_element;
#endif
    uint in_start_idx = buffer_offset + G;

    // The UINT_FAST32_MAX here is used to fill all 'coloumn' bits after n ...
    cols[L][d] = in_starts[in_start_idx].cols | (UINT_FAST32_MAX << N);
#ifdef ASSERT
    uint bitsset = popcount(cols[L][d]&COL_MASK);
    if((bitsset != (N-1))
    && (bitsset != (N-2))) {
        printf("[L] entry: wrong number of bits set: %d\n", bitsset);
    }
#endif
#ifdef DEBUG
    printf("L|IN  lid: %d, gid: %d, cols: %x, buf_off: %u, in_idx: %u, addr: %p\n",
           L, G, cols[L][d], buffer_offset, in_start_idx, &in_starts[in_start_idx]);
#endif
    // This places the first two queens
    diagl[L][d] = in_starts[in_start_idx].diagl;
    diagr[L][d] = in_starts[in_start_idx].diagr;

    in_starts[in_start_idx].cols = 0;
    in_starts[in_start_idx].diagl = 0;
    in_starts[in_start_idx].diagr = 0;

#ifdef LOOKAHEAD
//* TODO(sudden6): check if lookahead even works for last 2 steps
#define REST_INIT (N - LOOKAHEAD - PLACED)
#define STOP_DEPTH (REST_INIT - DEPTH + 1)

    // we're allready two rows into the field here
    rest[L][d] = REST_INIT;//*/
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
#ifdef LOOKAHEAD
      int_fast8_t l_rest = rest[L][d];
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
            uint_fast32_t lookahead1 = (bit | (new_diagl << (LOOKAHEAD - 2)) | (new_diagr >> (LOOKAHEAD - 2)));
            uint_fast8_t allowed1 = l_rest >= 0;

            if(allowed1 && (lookahead1 == UINT_FAST32_MAX)) {
                continue;
            }

            /*
            uint_fast32_t lookahead2 = (bit | (new_diagl << (LOOKAHEAD - 1)) | (new_diagr >> (LOOKAHEAD - 1)));
            uint_fast8_t allowed2 = l_rest > 0;

            if(allowed2 && (lookahead2 == UINT_FAST32_MAX)) {
                continue;
            }//*/

            l_rest--;
#endif
            //*/
            // The next two lines save stack depth + backtrack operations
            // when we passed the last possibility in a row.
            // Go lower in the stack, avoid branching by writing above the current
            // position
            posibs = posib;
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
#ifdef LOOKAHEAD
            rest[L][d] = l_rest;
#endif
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
      posib = posibs; // backtrack ...
      d--;
    }
    out_sum[G] += num;
}
