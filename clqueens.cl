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

#define MAX_EXPANSION (GPU_DEPTH)
#define SCRATCH_SIZE (WORKGROUP_SIZE * MAX_EXPANSION)
#define WORK_FACTOR 1

void solver_core_single(const __global start_condition* work_in, __local start_condition* scratch, __local uint* scratch_fill, uint lookahead_depth) {
	uint_fast32_t cols = work_in->cols;
    uint_fast32_t diagl = work_in->diagl;
    uint_fast32_t diagr = work_in->diagr;

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
            /* Optimizations from "Putting Queens in Carry Chains" {Thomas B. Preußer, Bernd Nägel, and Rainer G. Spallek} */

            // Page 8, 1)
            // Lookahead optimization, DEPTH 2
            uint_fast32_t lookahead2 = (bit | (new_diagl << 1) | (new_diagr >> 1));
            if ((lookahead_depth >= 2) && (lookahead2 == UINT_FAST32_MAX)) {
                continue;
            }

            // Lookahead optimization, DEPTH 3
            uint_fast32_t lookahead3 = (bit | (new_diagl << 2) | (new_diagr >> 2));
            if ((lookahead_depth >= 3) && (lookahead3 == UINT_FAST32_MAX)) {
                continue;
            }

#if 0
            // 2x2 square lookahead optimization Page 8, 2)
            if((lookahead_depth >= 3) && (popcount(~lookahead2) == 2)) {
                uint_fast32_t adjacent = ~lookahead2 & (~lookahead2 << 1);
                if (adjacent && ((lookahead2 == new_posib) || (lookahead2 == lookahead3))) {
                    continue;
                }
            }
#endif

#if 0
            // 2x2 square lookahead optimization Page 8, 2)
            if((lookahead_depth >= 3) && (lookahead2 == new_posib)) {
                uint_fast32_t adjacent = ~lookahead2 & (~lookahead2 << 1);
                if (adjacent && (popcount(~lookahead2) == 2)) {
                    continue;
                }
            }
#endif

            uint out_offs = atomic_add(scratch_fill, 1);
            //printf("found G: %u, out_offs: %u, bit: %x, diagl: %x, diagr: %x\n", G, out_offs, bit, new_diagl, new_diagr);

            scratch[out_offs].cols = bit;
            scratch[out_offs].diagr = new_diagr;
            scratch[out_offs].diagl = new_diagl;
        }
    }
}

kernel void solve_single(__global start_condition* workspace, __global uint* workspace_sizes, unsigned workspace_idx, unsigned factor) {

    __local start_condition scratch_buf[SCRATCH_SIZE * WORK_FACTOR];
    __local uint scratch_fill;

	if(L == 0) {
		scratch_fill = 0;
	}

    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    uint workspace_base_idx = workspace_idx * WORKSPACE_SIZE;

    // input index calculation
    uint in_first_idx = workspace_base_idx + workspace_sizes[workspace_idx] + (G - L)*factor;
    uint in_last_idx = in_first_idx + get_local_size(0) * factor;

    for(uint in_our_idx = in_first_idx + L; in_our_idx < in_last_idx; in_our_idx += get_local_size(0)) {
        solver_core_single(&workspace[in_our_idx], scratch_buf, &scratch_fill, 3);        
    }

    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    // output index calculation
    __local uint out_offs;
    uint out_first_idx = workspace_base_idx + WORKSPACE_SIZE;
    __global uint *out_cur_idx_ptr = workspace_sizes + workspace_idx + 1;

    if(L == 0) {
        uint out_fill = atomic_add(out_cur_idx_ptr, scratch_fill);
        out_offs = out_first_idx + out_fill;
    }

    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    for(int i = L; i < scratch_fill; i += get_local_size(0)) {
        workspace[out_offs + i] = scratch_buf[i];
        //printf("G: %u, L: %u, out_workspace_idx %u, cols: %x, diagl: %x, diagr: %x\n", G, L, out_offs + i,
        //       workspace[out_offs + i].cols, workspace[out_offs + i].diagl, workspace[out_offs + i].diagr);
    }
}

// need to duplicate here because scratch_buf size must be compile time constant
kernel void solve_pre_final(__global start_condition* workspace, __global uint* workspace_sizes, unsigned workspace_idx, unsigned factor) {

    __local start_condition scratch_buf[WORKGROUP_SIZE * 3 * WORK_FACTOR];
    __local uint scratch_fill;

	if(L == 0) {
		scratch_fill = 0;
	}

    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    uint workspace_base_idx = workspace_idx * WORKSPACE_SIZE;

    // input index calculation
    uint in_first_idx = workspace_base_idx + workspace_sizes[workspace_idx] + (G - L)*factor;
    uint in_last_idx = in_first_idx + get_local_size(0) * factor;

    for(uint in_our_idx = in_first_idx + L; in_our_idx < in_last_idx; in_our_idx += get_local_size(0)) {
        solver_core_single(&workspace[in_our_idx], scratch_buf, &scratch_fill, 2);        
    }

    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    // output index calculation
    __local uint out_offs;
    uint out_first_idx = workspace_base_idx + WORKSPACE_SIZE;
    __global uint *out_cur_idx_ptr = workspace_sizes + workspace_idx + 1;

    if(L == 0) {
        uint out_fill = atomic_add(out_cur_idx_ptr, scratch_fill);
        out_offs = out_first_idx + out_fill;
    }

    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    for(int i = L; i < scratch_fill; i += get_local_size(0)) {
        workspace[out_offs + i] = scratch_buf[i];
        //printf("G: %u, L: %u, out_workspace_idx %u, cols: %x, diagl: %x, diagr: %x\n", G, L, out_offs + i,
        //       workspace[out_offs + i].cols, workspace[out_offs + i].diagl, workspace[out_offs + i].diagr);
    }
}

kernel void solve_final(__global start_condition* workspace, __global uint* workspace_sizes, __global uint* out_res, unsigned factor) {
	__local start_condition scratch_buf[WORKGROUP_SIZE * 2 * WORK_FACTOR];
    __local uint scratch_fill;

	if(L == 0) {
		scratch_fill = 0;
	}

    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    const uint workspace_idx = GPU_DEPTH - 2;
    uint workspace_base_idx = workspace_idx * WORKSPACE_SIZE;

    // input index calculation
    uint in_first_idx = workspace_base_idx + workspace_sizes[workspace_idx] + (G - L)*factor;
    uint in_last_idx = in_first_idx + get_local_size(0) * factor;

    for(uint in_our_idx = in_first_idx + L; in_our_idx < in_last_idx; in_our_idx += get_local_size(0)) {
		 solver_core_single(&workspace[in_our_idx], scratch_buf, &scratch_fill, 1); 
	}

	work_group_barrier(CLK_LOCAL_MEM_FENCE);
    //printf("G: %u, L: %u, L_size: %u, in_our_idx: %u", G, L, get_local_size(0), in_our_idx);

    uint cnt = 0;
	for(int i = L; i < scratch_fill; i += get_local_size(0)) {
        uint_fast32_t cols = scratch_buf[i].cols;
        // This places the first two queens
        uint_fast32_t diagl = scratch_buf[i].diagl;
        uint_fast32_t diagr = scratch_buf[i].diagr;

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

    uint limits[GPU_DEPTH - 1];

    uint even_launches = 0;
    uint odd_launches = 0;

    // final run only depends on input limit, calculate differently
    limits[GPU_DEPTH-2] = workspace_sizes[GPU_DEPTH - 2];

    // Find maximum possible kernel launches
    //printf("placed: 0x%x, recursion: %u\n", placed, recursion);
    for(uint i = 0; i < (GPU_DEPTH - 2); i++) {
        uint input_limit = workspace_sizes[i];
        uint output_limit = (WORKSPACE_SIZE - workspace_sizes[i+1]) / expansion_factor(i + FIRST_PLACED);
        uint limit = min(input_limit, output_limit);
        //printf("  size[%u] = %u, output_limit: %u\n",i, workspace_sizes[i], output_limit);
        limits[i] = limit;
    }

    //printf("  size[%u] = %u\n", GPU_DEPTH-1, workspace_sizes[GPU_DEPTH-1]);

    for(uint i = 0; i < (GPU_DEPTH - 1); i++) {
        if (i % 2) {
            // odd
            odd_launches += limits[i];
        } else {
            // even
            even_launches += limits[i];
        }
    }

    if (odd_launches == 0 && even_launches == 0) {
        printf("Finished, recursion: %u\n", recursion);
        return;
    }

#if 1
    // An unlimited recursion level potentially allows to fully use the GPU queue,
    // but this might be a corner case in the code and result in problems
    if (recursion > 200) {
        //printf("Recursion limit\n");
        return;
    }
#endif

#if 0
    if (state == CLSOLVER_FEED && workspace_sizes[0] < WORKSPACE_SIZE/(MAX_EXPANSION)) {
        //printf("Re-feed, recursion: %u\n", recursion);
        // re-feed
        return;
    }
#endif

    uint launch_odd = odd_launches > even_launches;

    //printf("Launch odd: %u\n", launch_odd);

    int err = 0;
    
    clk_event_t launched_kernels_evt[GPU_DEPTH-1];
    uint launched_kernels_cnt = 0;

    for(uint workspace_idx = launch_odd; workspace_idx < (GPU_DEPTH-1); workspace_idx += 2) {
        uint applied_factor = 1;
        uint launch_cnt = limits[workspace_idx];

        if (launch_cnt == 0) {
            //printf("Skipping, workspace_idx: %u\n", workspace_idx);
            continue;
        }

        if (launch_cnt > WORKSPACE_SIZE) {
            printf("Corruption detected\n");
            return;
        }

#if 1
        // Align to workgroup size if possible
        if(launch_cnt > WORKGROUP_SIZE) {
            uint rest = launch_cnt % WORKGROUP_SIZE;
            launch_cnt -= rest;
        }
#endif

        // Align to WORK_FACTOR
        if(launch_cnt > WORK_FACTOR) {
            uint rest = launch_cnt % WORK_FACTOR;
            launch_cnt -= rest;
            applied_factor = WORK_FACTOR;
        }

        void (^solve_final_blk)(void) = ^{
                    solve_final(workspace, workspace_sizes, out_res, applied_factor);
                };
        
        void (^solve_pre_final_blk)(void) = ^{
                    solve_pre_final(workspace, workspace_sizes, workspace_idx, applied_factor);
                };
        
        void (^solve_single_blk)(void) = ^{
                    solve_single(workspace, workspace_sizes, workspace_idx, applied_factor);
                };
        
        // HACK: must init block at declaration
        void (^run_blk)(void) = workspace_idx == (GPU_DEPTH - 3) ? solve_pre_final_blk :
                                (workspace_idx == (GPU_DEPTH - 2) ? solve_final_blk : solve_single_blk);
        uint local_size = min((uint)WORKGROUP_SIZE, (uint)get_kernel_work_group_size(run_blk));
        //printf("local_size: %u, kernel_wg_size: %u\n", local_size, get_kernel_work_group_size(run_blk));

        //printf("launch workspace_idx: %u, launch_cnt: %u, factor: %u, local_size: %u\n", workspace_idx, launch_cnt, applied_factor, local_size);

        // launch kernel
        // must wait after this one finishes, because of write to 'workspace_sizes'
        err = enqueue_kernel(q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
                            ndrange_1D(launch_cnt/applied_factor, local_size),
                            0,
                            NULL,
                            &launched_kernels_evt[launched_kernels_cnt],
                            run_blk);

        if (err != 0) {
            printf("Error when enqueuing kernel, launch_cnt: %u, workspace_idx: %u, state: 0x%x", launch_cnt, workspace_idx, state);
            goto cleanup_kernels_evt;
        } else  {
            //printf("launch removed: %u", max_launches);
            // remove completed work items only when successfully launched
            workspace_sizes[workspace_idx] -= launch_cnt;
            launched_kernels_cnt++;
        }
    }

    //printf("launched_kernels_cnt: %u\n", launched_kernels_cnt);

    clk_event_t marker_evt;
    err = enqueue_marker(q, launched_kernels_cnt, launched_kernels_evt, &marker_evt);
    if ( err != 0) {
        printf("Error enqueuing marker\n");
        goto cleanup_kernels_evt;
    }

    void (^recursion_blk)(void) = ^{
                    relaunch_kernel(workspace, workspace_sizes, out_res, placed, recursion + 1);
                };

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

    for(uint i = 0; i < launched_kernels_cnt; i++) {
        release_event(launched_kernels_evt[i]);
    }

    release_event(marker_evt);
    return;

    cleanup_kernels_evt:
    for(uint i = 0; i < launched_kernels_cnt; i++) {
        release_event(launched_kernels_evt[i]);
    }
}
