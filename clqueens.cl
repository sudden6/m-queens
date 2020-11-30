// SYNC: Keep in sync with solverstructs.h
struct __attribute__ ((packed)) start_condition_t {
    uint cols; // bitfield with all the used columns
    uint diagl;// bitfield with all the used diagonals down left
    uint diagr;// bitfield with all the used diagonals down right
};

typedef struct start_condition_t start_condition;

#define CLSOLVER_STATE_MASK (1U << 31)
#define CLSOLVER_FEED (0)
#define CLSOLVER_CLEANUP CLSOLVER_STATE_MASK

// END SYNC

//#define printf(...)

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

uint expansion_factor(uint placed) {
    return BOARDSIZE - placed;
}

// TODO: replace with compiler defined values
//#define WORKSPACE_SIZE 16
//#define GPU_DEPTH 2
//#define BOARDSIZE 8

#define LAST_PLACED (BOARDSIZE - 1)
#define FIRST_PLACED (BOARDSIZE - GPU_DEPTH)

uint get_tmp_idx(uint placed) {
    return placed - FIRST_PLACED;
}

#define MAX_EXPANSION ((FIRST_PLACED - 1))
#define SCRATCH_SIZE (WORKGROUP_SIZE * MAX_EXPANSION)

kernel void relaunch_kernel(__global start_condition* workspace, __global uint* workspace_sizes, __global uint* out_res, unsigned placed, unsigned recursion);

// without lookahead optimization
kernel void solve_single_no_look(__global start_condition* workspace, __global uint* workspace_sizes, __global uint* out_res, unsigned placed_state, unsigned recursion) {

    __local start_condition scratch_buf[SCRATCH_SIZE];
    __local uint scratch_fill;

    uint placed = placed_state & ~CLSOLVER_STATE_MASK;

	if(L == 0) {
		scratch_fill = 0;
	}

    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    // input index calculation
    uint in_first_idx = get_tmp_idx(placed) * WORKSPACE_SIZE;

    // elements are subtracted from size by relaunch kernel
    uint in_our_idx = in_first_idx + workspace_sizes[get_tmp_idx(placed)] + G;

    // algorithm start
    uint_fast32_t cols = workspace[in_our_idx].cols;
    uint_fast32_t diagl = workspace[in_our_idx].diagl;
    uint_fast32_t diagr = workspace[in_our_idx].diagr;

    //printf("solve G: %u, in_our_idx: %u, cols: %x, diagl: %x, diagr: %x\n", G, in_our_idx, cols, diagl, diagr);

    //  The variable posib contains the bitmask of possibilities we still have
    //  to try in a given row ...
    uint_fast32_t posib = (cols | diagl | diagr);

    diagl <<= 1;
    diagr >>= 1;

    while (posib != UINT_FAST32_MAX) {
        // The standard trick for getting the rightmost bit in the mask
        uint_fast32_t bit = ~posib & (posib + 1);
        posib ^= bit; // Eliminate the tried possibility.
        uint_fast32_t new_diagl = (bit << 1) | diagl;
        uint_fast32_t new_diagr = (bit >> 1) | diagr;
        bit |= cols;
        uint_fast32_t new_posib = (bit | new_diagl | new_diagr);

        if (new_posib != UINT_FAST32_MAX) {
            uint out_offs = atomic_add(&scratch_fill, 1);
            //printf("G: %u, out_idx: %u, bit: %x, diagl: %x, diagr: %x\n", G, out_idx, bit, diagl, diagr);

            scratch_buf[out_offs].cols = bit;
            scratch_buf[out_offs].diagr = new_diagr;
            scratch_buf[out_offs].diagl = new_diagl;
        }
    }

    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    uint out_fill = 0;
    // output index calculation
    uint out_first_idx = get_tmp_idx(placed + 1) * WORKSPACE_SIZE;

    if(L == 0) {
        __global uint *out_cur_idx_ptr = workspace_sizes + get_tmp_idx(placed + 1);
        out_fill = atomic_add(out_cur_idx_ptr, scratch_fill);
    }

    out_fill = work_group_broadcast(out_fill, 0);
    uint out_offs = out_first_idx + out_fill;

    //printf("G: %u, out_fill: %u, out_offs: %u, L: %u\n", G, out_fill, out_offs, L);

    for(int i = L; i < scratch_fill; i += get_local_size(0)) {
        workspace[out_offs + i] = scratch_buf[i];
    }


    // launch relaunch_kernel
    if(G == 0) {
        queue_t q = get_default_queue();
        int err = enqueue_kernel(q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
                                ndrange_1D(1),
                                ^{
                                    relaunch_kernel(workspace, workspace_sizes, out_res, placed_state, recursion + 1);
                                });
        if (err != 0) {
            printf("Error when enqueuing from solver kernel");
        }
    }
}

kernel void solve_final(__global start_condition* workspace, __global uint* workspace_sizes, __global uint* out_res, unsigned placed_state, unsigned recursion, unsigned factor) {
    // input index calculation
    uint in_first_idx = get_tmp_idx(LAST_PLACED) * WORKSPACE_SIZE;
    uint G_first = (G - L) * factor;
    uint in_last_idx = in_first_idx + G_first + get_local_size(0) * factor;
    uint cnt = 0;

    for(uint in_our_idx = in_first_idx + G_first + L; in_our_idx < in_last_idx; in_our_idx += get_local_size(0)) {
    //printf("G: %u, L: %u, L_size: %u, in_our_idx: %u", G, L, get_local_size(0), in_our_idx);

        uint_fast32_t cols = workspace[in_our_idx].cols;
        // This places the first two queens
        uint_fast32_t diagl = workspace[in_our_idx].diagl;
        uint_fast32_t diagr = workspace[in_our_idx].diagr;

        //  The variable posib contains the bitmask of possibilities we still have
        //  to try in a given row ...
        uint_fast32_t posib = (cols | diagl | diagr);

        diagl <<= 1;
        diagr >>= 1;

        //printf("G: %u, posib: %x", G, posib);

        if (posib != UINT_FAST32_MAX) {
            // The standard trick for getting the rightmost bit in the mask
            uint_fast32_t bit = ~posib & (posib + 1);
            posib ^= bit; // Eliminate the tried possibility.
            uint_fast32_t new_diagl = (bit << 1) | diagl;
            uint_fast32_t new_diagr = (bit >> 1) | diagr;
            bit |= cols;
            uint_fast32_t new_posib = (bit | new_diagl | new_diagr);

            cnt += new_posib == UINT_FAST32_MAX && bit == UINT_FAST32_MAX;
        }
    }

    //printf("G: %u, cnt: %u", G, cnt);

    out_res[G] += cnt;
    // launch relaunch_kernel
    if(G == 0) {
        queue_t q = get_default_queue();
        int err = enqueue_kernel(q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
                      ndrange_1D(1),
                      ^{
                           relaunch_kernel(workspace, workspace_sizes, out_res, placed_state, recursion + 1);
                       });
        
        if (err != 0) {
            printf("Error when enqueuing from final kernel");
        }
    }
}


kernel void relaunch_kernel(__global start_condition* workspace, __global uint* workspace_sizes, __global uint* out_res, unsigned placed, unsigned recursion) {
    queue_t q = get_default_queue();
    uint next_placed = 0;
    // limit due to input start conditions
    uint max_launches = 0;
    uint state = placed & CLSOLVER_STATE_MASK;

    // Find maximum possible kernel launches
    //printf("placed: 0x%x, recursion: %u\n", placed, recursion);
    for(uint i = 0; i < (GPU_DEPTH - 1); i++) {
        uint input_limit = workspace_sizes[i];
        uint l_placed = i + FIRST_PLACED;
        uint output_limit = (WORKSPACE_SIZE - workspace_sizes[i+1]) / expansion_factor(l_placed);
        uint limit = min(input_limit, output_limit);
        //printf("  size[%u] = %u, output_limit: %u\n",i, workspace_sizes[i], output_limit);


        if(limit > max_launches) {
            max_launches = limit;
            next_placed = l_placed;
        }
    }

    // final run only depends on input limit, calculate extra
    uint input_limit = workspace_sizes[GPU_DEPTH - 1];
    //printf("  size[%u] = %u\n", GPU_DEPTH - 1, workspace_sizes[GPU_DEPTH - 1]);
    if (input_limit > max_launches) {
        max_launches = input_limit;
        next_placed = LAST_PLACED;
    }

    if (max_launches == 0) {
        printf("Finished, recursion: %u\n", recursion);
        return;
    }

#if 0
    if (recursion > 1000) {
        printf("Recursion limit, next_placed: %u\n", next_placed);
        return;
    }
#endif

#if 0
    if (state == CLSOLVER_FEED && workspace_sizes[0] < WORKSPACE_SIZE/(MAX_EXPANSION)) {
        printf("Re-feed, recursion: %u\n", recursion);
        // re-feed
        return;
    }
#endif



    int err = 0;
    ndrange_t range = ndrange_1D(max_launches, WORKGROUP_SIZE);
    uint factor = 128;

    if(next_placed == LAST_PLACED) {
        // last step, count solutions
        //printf("Last step, launched: %u\n", max_launches);
        uint applied_factor = 1;

        if(max_launches > factor) {
            uint rest = max_launches % factor;
            max_launches -= rest;
            range = ndrange_1D(max_launches/factor, WORKGROUP_SIZE);
            applied_factor = factor;
            //printf("max_launches: %u, applied_factor: %u\n", max_launches, applied_factor);
        }

        // launch kernel
        err = enqueue_kernel(q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
                            range,
                            ^{
                                solve_final(workspace, workspace_sizes, out_res, next_placed | state, recursion + 1, applied_factor);
                            });
    } else {
        // Intermediate step
        //printf("Single step, next_placed: %u, launched: %u\n", next_placed, max_launches);
        // launch kernel
        err = enqueue_kernel(q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
                            range,
                            ^{
                                solve_single_no_look(workspace, workspace_sizes, out_res, next_placed | state, recursion + 1);
                            });
    }

    if (err != 0) {
        printf("Error when enqueuing kernel, launches: %u, next_placed: %u, state: 0x%x, recursion: %u", max_launches, next_placed, state, recursion);
    } else  {
        //printf("launch removed: %u", max_launches);
        // remove completed work items only when successfully launched
        workspace_sizes[get_tmp_idx(next_placed)] -= max_launches;
    }
}
