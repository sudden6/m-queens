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
#define WORK_FACTOR 1

// without lookahead optimization
kernel void solve_single_no_look(__global start_condition* workspace, __global uint* workspace_sizes, __global uint* out_res, unsigned placed, unsigned factor) {

    __local start_condition scratch_buf[SCRATCH_SIZE * WORK_FACTOR];
    __local uint scratch_fill;

	if(L == 0) {
		scratch_fill = 0;
	}

    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    // input index calculation
    uint in_first_idx = get_tmp_idx(placed) * WORKSPACE_SIZE + workspace_sizes[get_tmp_idx(placed)] + (G - L)*factor;
    uint in_last_idx = in_first_idx + get_local_size(0) * factor;

    for(uint in_our_idx = in_first_idx + L; in_our_idx < in_last_idx; in_our_idx += get_local_size(0)) {

        // algorithm start
        uint_fast32_t cols = workspace[in_our_idx].cols;
        uint_fast32_t diagl = workspace[in_our_idx].diagl;
        uint_fast32_t diagr = workspace[in_our_idx].diagr;

        //printf("solve G: %u, L: %u, in_our_idx: %u, cols: %x, diagl: %x, diagr: %x\n", G, L, in_our_idx, cols, diagl, diagr);

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
    }

    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    // output index calculation
    uint out_first_idx = get_tmp_idx(placed + 1) * WORKSPACE_SIZE;
    __local uint out_offs;

    if(L == 0) {
        //printf("G: %u, L: %u, scratch_fill: %u\n", G, L, scratch_fill);
        __global uint *out_cur_idx_ptr = workspace_sizes + get_tmp_idx(placed + 1);
        uint out_fill = atomic_add(out_cur_idx_ptr, scratch_fill);
        out_offs = out_first_idx + out_fill;
    }

    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    //printf("G: %u, L: %u, out_fill: %u, out_offs: %u\n", G, L, out_fill, out_offs);

    for(int i = L; i < scratch_fill; i += get_local_size(0)) {
        workspace[out_offs + i] = scratch_buf[i];
    }
}

kernel void solve_final(__global start_condition* workspace, __global uint* workspace_sizes, __global uint* out_res, unsigned factor) {
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
}


kernel void relaunch_kernel(__global start_condition* workspace, __global uint* workspace_sizes, __global uint* out_res, unsigned placed, unsigned recursion) {
    queue_t q = get_default_queue();
    uint state = placed & CLSOLVER_STATE_MASK;

    uint limits[GPU_DEPTH];

    uint even_launches = 0;
    uint odd_launches = 0;

    // final run only depends on input limit, calculate differently
    limits[GPU_DEPTH-1] = workspace_sizes[GPU_DEPTH - 1];

    // Find maximum possible kernel launches
    //printf("placed: 0x%x, recursion: %u\n", placed, recursion);
    for(uint i = 0; i < (GPU_DEPTH - 1); i++) {
        uint input_limit = workspace_sizes[i];
        uint output_limit = (WORKSPACE_SIZE - workspace_sizes[i+1]) / expansion_factor(i + FIRST_PLACED);
        uint limit = min(input_limit, output_limit);
        //printf("  size[%u] = %u, output_limit: %u\n",i, workspace_sizes[i], output_limit);
        limits[i] = limit;
    }

    //printf("  size[%u] = %u\n", GPU_DEPTH-1, workspace_sizes[GPU_DEPTH-1]);

    for(uint i = 0; i < GPU_DEPTH; i++) {
        if (i % 2) {
            // odd
            odd_launches += limits[i];
        } else {
            // even
            even_launches += limits[i];
        }
    }

    //printf("  size[%u] = %u\n", GPU_DEPTH - 1, workspace_sizes[GPU_DEPTH - 1]);

    if (odd_launches == 0 && even_launches == 0) {
        printf("Finished, recursion: %u\n", recursion);
        return;
    }

    if (recursion > 3) {
        //printf("Recursion limit\n");
        return;
    }

#if 0
    if (state == CLSOLVER_FEED && workspace_sizes[0] < WORKSPACE_SIZE/(MAX_EXPANSION)) {
        printf("Re-feed, recursion\n");
        // re-feed
        return;
    }
#endif

    uint launch_odd = odd_launches > even_launches;

    //printf("Launch odd: %u\n", launch_odd);

    int err = 0;
    
    clk_event_t launched_kernels_evt[GPU_DEPTH];
    uint launched_kernels_cnt = 0;

    for(uint workspace_idx = launch_odd; workspace_idx < GPU_DEPTH; workspace_idx += 2) {
        uint next_placed = workspace_idx + FIRST_PLACED;
        uint applied_factor = 1;
        uint launch_cnt = limits[workspace_idx];
        if(launch_cnt > WORK_FACTOR) {
                uint rest = launch_cnt % WORK_FACTOR;
                launch_cnt -= rest;
                applied_factor = WORK_FACTOR;
                //printf("max_launches: %u, applied_factor: %u\n", max_launches, applied_factor);
        }

        void (^solve_final_blk)(void) = ^{
                    solve_final(workspace, workspace_sizes, out_res, applied_factor);
                };
        
        void (^solve_single_blk)(void) = ^{
                    solve_single_no_look(workspace, workspace_sizes, out_res, next_placed, applied_factor);
                };
      
        void (^run_blk)(void) = next_placed == LAST_PLACED ? solve_final_blk : solve_single_blk;
        uint local_size = min((uint)WORKGROUP_SIZE, get_kernel_work_group_size(run_blk));

        // launch kernel
        err = enqueue_kernel(q, CLK_ENQUEUE_FLAGS_NO_WAIT,
                            ndrange_1D(launch_cnt/applied_factor, local_size),
                            0,
                            NULL,
                            &launched_kernels_evt[launched_kernels_cnt],
                            run_blk);

        if(next_placed == LAST_PLACED) {
            // last step, count solutions
            //printf("Last step, launched: %u\n", max_launches);
        } else {
            // Intermediate step
            //printf("Single step, next_placed: %u, launched: %u\n", next_placed, max_launches);
        }

        if (err != 0) {
            printf("Error when enqueuing kernel, launches: %u, next_placed: %u, state: 0x%x", launch_cnt, next_placed, state);
            goto cleanup_kernels_evt;
        } else  {
            //printf("launch removed: %u", max_launches);
            // remove completed work items only when successfully launched
            workspace_sizes[workspace_idx] -= launch_cnt;
            launched_kernels_cnt++;
        }
    }

    clk_event_t marker_evt;
    err = enqueue_marker(q, launched_kernels_cnt, launched_kernels_evt, &marker_evt);
    if ( err != 0) {
        printf("Error enqueuing marker\n");
        goto cleanup_kernels_evt;
    }

    void (^recursion_blk)(void) = ^{
                    relaunch_kernel(workspace, workspace_sizes, out_res, placed, recursion + 1);
                };

#if 0
    // self enqueue recursion
    err = enqueue_kernel(q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
                        ndrange_1D(1, 1),
                        1,
                        &marker_evt,
                        NULL,
                        recursion_blk);
    if ( err != 0) {
        printf("Error enqueuing recursion: %u\n", recursion);
    }
#endif

    release_event(marker_evt);

    cleanup_kernels_evt:
    for(uint i = 0; i < launched_kernels_cnt; i++) {
        release_event(launched_kernels_evt[i]);
    }
}
