typedef struct __attribute__ ((packed)) {
    uint cols; // bitfield with all the used columns
    uint diagl;// bitfield with all the used diagonals down left
    uint diagr;// bitfield with all the used diagonals down right
} start_condition;

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
    return placed - 1;
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

kernel void relaunch_kernel(__global start_condition* workspace, __global uint* workspace_sizes, __global uint* out_res, unsigned placed);

// without lookahead optimization
kernel void solve_single_no_look(__global start_condition* workspace, __global uint* workspace_sizes, __global uint* out_res, unsigned placed) {

    __local start_condition scratch_buf[SCRATCH_SIZE];
    __local uint scratch_fill;

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
                                    relaunch_kernel(workspace, workspace_sizes, out_res, placed);
                                });
        if (err != 0) {
            printf("Error when enqueuing solver kernel");
        }
    }
}

kernel void solve_final(__global start_condition* workspace, __global uint* workspace_sizes, __global uint* out_res) {
    // input index calculation
    uint in_first_idx = get_tmp_idx(LAST_PLACED) * WORKSPACE_SIZE;
    uint in_our_idx = in_first_idx + G;

    //printf("G: %u, in_our_idx: %u", G, in_our_idx);

    uint cnt = 0;

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

    while (posib != UINT_FAST32_MAX) {
        // The standard trick for getting the rightmost bit in the mask
        uint_fast32_t bit = ~posib & (posib + 1);
        posib ^= bit; // Eliminate the tried possibility.
        uint_fast32_t new_diagl = (bit << 1) | diagl;
        uint_fast32_t new_diagr = (bit >> 1) | diagr;
        bit |= cols;
        uint_fast32_t new_posib = (bit | new_diagl | new_diagr);

        cnt += new_posib == UINT_FAST32_MAX && bit == UINT_FAST32_MAX;
    }

    //printf("G: %u, cnt: %u", G, cnt);

    out_res[G] += cnt;
    // launch relaunch_kernel
    if(G == 0) {
        queue_t q = get_default_queue();
        int err = enqueue_kernel(q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
                      ndrange_1D(1),
                      ^{
                           relaunch_kernel(workspace, workspace_sizes, out_res, FIRST_PLACED);
                       });
        
        if (err != 0) {
            printf("Error when enqueuing final kernel");
        }
    }
}


kernel void relaunch_kernel(__global start_condition* workspace, __global uint* workspace_sizes, __global uint* out_res, unsigned placed) {
    queue_t q = get_default_queue();
    uint next_placed = 0;
    // limit due to input start conditions
    uint max_launches = 0;

    // Find maximum possible kernel launches
    //printf("placed: %u\n", placed);
    for(uint i = 0; i < (GPU_DEPTH - 1); i++) {
        //printf("  size[%u] = %u\n",i, workspace_sizes[i]);
        uint input_limit = workspace_sizes[i];
        uint l_placed = i + FIRST_PLACED;
        uint output_limit = (WORKSPACE_SIZE - workspace_sizes[i+1]) / expansion_factor(l_placed);
        uint limit = min(input_limit, output_limit);

        if(limit > max_launches) {
            max_launches = limit;
            next_placed = l_placed;
        }
    }

    // final run only depends on input limit, calculate extra
    uint input_limit = workspace_sizes[GPU_DEPTH - 1];
    if (input_limit > max_launches) {
        max_launches = input_limit;
        next_placed = LAST_PLACED;
    }

#if 0
    if ((max_launches < WORKSPACE_SIZE/32) && workspace_sizes[0] > WORKSPACE_SIZE / 2) {
        // re-feed
        return;
    }
#endif

    if (max_launches == 0) {
        //printf("Finished\n");
        return;
    }

    int err = 0;
    ndrange_t range = ndrange_1D(max_launches, get_local_size(0));

    if(next_placed == LAST_PLACED) {
        // last step, count solutions
        //printf("Last step, launched: %u\n", max_launches);

        // launch kernel
        err = enqueue_kernel(q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
                            range,
                            ^{
                                solve_final(workspace, workspace_sizes, out_res);
                            });
    } else {
        // Intermediate step
        //printf("Single step, next_placed: %u, launched: %u\n", next_placed, max_launches);
        // launch kernel
        err = enqueue_kernel(q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
                            range,
                            ^{
                                solve_single_no_look(workspace, workspace_sizes, out_res, next_placed);
                            });
    }

    if (err != 0) {
        printf("Error when enqueuing kernel, launches: %u, next_placed: %u", max_launches, next_placed);
    } else  {
        //printf("launch removed: %u", max_launches);
        // remove completed work items only when successfully launched
        workspace_sizes[get_tmp_idx(next_placed)] -= max_launches;
    }
}
