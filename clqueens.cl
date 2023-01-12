// SYNC: Keep in sync with solverstructs.h
struct __attribute__ ((packed)) start_condition_t {
    uint cols; // bitfield with all the used columns
    uint diagl;// bitfield with all the used diagonals down left
    uint diagr;// bitfield with all the used diagonals down right
};

typedef struct start_condition_t start_condition;

#define CLSOLVER_FEED (0)
#define CLSOLVER_CLEANUP (1)

// END SYNC

// For analysis tools
#ifndef BOARDSIZE
#define BOARDSIZE 20
#define GPU_DEPTH 11
#define WORKSPACE_SIZE 33554432
#define SUM_REDUCTION_FACTOR 32768
#define WORKGROUP_SIZE 64
#endif

#if DEBUG_PRINT == 0
#define printf(...)
#endif

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

// TODO: replace with compiler defined values
//#define WORKSPACE_SIZE 16
//#define GPU_DEPTH 2
//#define BOARDSIZE 8

#define MAX_EXPANSION (GPU_DEPTH)
#define SCRATCH_SIZE (WORKGROUP_SIZE * MAX_EXPANSION)
#define WORK_FACTOR 8

inline void solver_core_single(const __global start_condition* work_in, volatile __local start_condition* scratch, volatile __local atomic_uint* scratch_fill, uint lookahead_depth) {
    uint_fast32_t cols = work_in->cols;
    uint_fast32_t diagl = work_in->diagl;
    uint_fast32_t diagr = work_in->diagr;

    //printf("[core ] solve G: %lu, L: %lu, cols: %x, diagl: %x, diagr: %x\n", G, L, cols, diagl, diagr);

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
            if ((lookahead_depth > 2) && (lookahead2 == UINT_FAST32_MAX)) {
                continue;
            }

            // Lookahead optimization, DEPTH 3
            uint_fast32_t lookahead3 = (bit | (new_diagl << 2) | (new_diagr >> 2));
            if ((lookahead_depth > 3) && (lookahead3 == UINT_FAST32_MAX)) {
                continue;
            }

#if 0
            // 2x2 square lookahead optimization Page 8, 2)
            if((lookahead_depth > 3) && (popcount(~lookahead2) == 2)) {
                uint_fast32_t adjacent = ~lookahead2 & (~lookahead2 << 1);
                if (adjacent && ((lookahead2 == new_posib) || (lookahead2 == lookahead3))) {
                    continue;
                }
            }
#endif

#if 0
            // 2x2 square lookahead optimization Page 8, 2)
            if((lookahead_depth > 3) && (lookahead2 == new_posib)) {
                uint_fast32_t adjacent = ~lookahead2 & (~lookahead2 << 1);
                if (adjacent && (popcount(~lookahead2) == 2)) {
                    continue;
                }
            }
#endif

            uint out_offs = atomic_fetch_add(scratch_fill, 1);
            //printf("[core ] found G: %lu, out_offs: %u, bit: %x, diagl: %x, diagr: %x\n", G, out_offs, bit, new_diagl, new_diagr);

            scratch[out_offs].cols = bit;
            scratch[out_offs].diagr = new_diagr;
            scratch[out_offs].diagl = new_diagl;
        }
    }
}

inline void solve_single_proto(const __global start_condition* in_start,
                        __global start_condition* out_base,
                        volatile __global atomic_uint* out_cur_offs,
                        volatile __local start_condition* scratch_buf,
                        unsigned factor, volatile __local atomic_uint* scratch_fill, uint remaining) {

    uint stride = get_local_size(0);
    uint offs = (G-L)*factor + L;
    uint end = offs + stride*factor;

    for(uint in_our_idx = offs; in_our_idx < end; in_our_idx += stride) {
        //printf("[pr %2u] G: %lu, L: %lu, L_size: %lu, in_our_idx: %u\n", remaining, G, L, get_local_size(0), in_our_idx);
        solver_core_single(&in_start[in_our_idx], scratch_buf, scratch_fill, remaining);
        // Wait until all threads added their results to the scratch buffer
        work_group_barrier(CLK_LOCAL_MEM_FENCE);
        // Retrieve fill level for all threads before we destroy it
        uint l_scratch_fill = atomic_load(scratch_fill);
        // Wait until all threads received the value
        work_group_barrier(CLK_LOCAL_MEM_FENCE);
        uint out_offs = 0;

        if(L == 0) {
            // Reserve space for results in global buffer only once
            out_offs = atomic_fetch_add(out_cur_offs, l_scratch_fill);
            // Zero scratch buffer fill level only once
            atomic_store(scratch_fill, 0);
        }

        // Make global buffer offset known to all threads
        out_offs = work_group_broadcast(out_offs, 0);

        // Copy scratch buffer contents to global buffer using all threads
        for(uint i = L; i < l_scratch_fill; i += get_local_size(0)) {
            out_base[out_offs + i] = scratch_buf[i];
            //printf("[pr %2u] G: %lu, L: %lu, out_workspace_idx %u, cols: %x, diagl: %x, diagr: %x\n", remaining, G, L, out_offs + i,
            //       out_base[out_offs + i].cols,out_base[out_offs + i].diagl, out_base[out_offs + i].diagr);
        }

        // Wait until copy for all threads finished, so that we don't overwrite the scratch buffer before it's copied
        work_group_barrier(CLK_LOCAL_MEM_FENCE);
    }
}

kernel void solve_single(const __global start_condition* in_start, __global start_condition* out_base,
                         volatile __global uint* out_cur_offs, unsigned factor) {
    volatile __local start_condition scratch_buf[SCRATCH_SIZE];
    volatile __local atomic_uint scratch_fill;

    if (L == 0) {
        atomic_init(&scratch_fill, 0);
    }

    // Wait until thread 0 initialized the atomic variables
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    solve_single_proto(in_start, out_base, (volatile __global atomic_uint*) out_cur_offs, scratch_buf, factor, &scratch_fill, 11);
}

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

#define SOLVER_KERNEL(remaining) \
kernel void solve_##remaining (const __global start_condition* in_start, __global start_condition* out_base, __global uint* out_cur_offs, unsigned factor) { \
    volatile __local start_condition scratch_buf[WORKGROUP_SIZE * (remaining)]; \
    volatile __local atomic_uint scratch_fill; \
    if (L == 0) { \
        atomic_init(&scratch_fill, 0); \
    } \
    work_group_barrier(CLK_LOCAL_MEM_FENCE); \
    solve_single_proto(in_start, out_base, (volatile __global atomic_uint*) out_cur_offs, scratch_buf, factor, &scratch_fill, (remaining));  \
}

SOLVER_KERNEL(3)
SOLVER_KERNEL(4)
SOLVER_KERNEL(5)
SOLVER_KERNEL(6)
SOLVER_KERNEL(7)
SOLVER_KERNEL(8)
SOLVER_KERNEL(9)
SOLVER_KERNEL(10)

kernel void solve_final(const __global start_condition* in_start, __global ulong* out_res, unsigned factor) {
    __local start_condition scratch_buf[WORKGROUP_SIZE * 2];
    __local atomic_uint scratch_fill;
    __local atomic_uint scratch_cnt;

    if (L == 0) {
        atomic_init(&scratch_fill, 0);
        atomic_init(&scratch_cnt, 0);
    }

    // Wait until thread 0 initialized the atomic variables
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    uint stride = get_local_size(0);
    uint offs = (G-L)*factor + L;
    uint end = offs + stride*factor;

    uint cnt = 0;
    for(uint in_our_idx = offs; in_our_idx < end; in_our_idx += stride) {

        //printf("[final] G: %lu, L: %lu, L_size: %lu, in_our_idx: %u\n", G, L, get_local_size(0), in_our_idx);
        //work_group_barrier(CLK_LOCAL_MEM_FENCE);
        solver_core_single(&in_start[in_our_idx], scratch_buf, &scratch_fill, 2);

        // Wait until all threads added their results to the scratch buffer
        work_group_barrier(CLK_LOCAL_MEM_FENCE);

        // Retrieve fill level for all threads before we destroy it
        const uint l_scratch_fill = atomic_load(&scratch_fill);

        // Wait until all threads retrieved the fill level
        work_group_barrier(CLK_LOCAL_MEM_FENCE);

        // Reset fill level only once using thread 0
        if (L == 0) {
            atomic_store(&scratch_fill, 0);
        }

        // Wait until fill level is reset
        work_group_barrier(CLK_LOCAL_MEM_FENCE);

        for(int i = L; i < l_scratch_fill; i += get_local_size(0)) {
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
    }

    // Add the result for each thread to scratch_cnt
    atomic_fetch_add(&scratch_cnt, cnt);

    // Wait until all threads added their result count
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    printf("[final] G: %lu, cnt: %u\n", G, cnt);
    if (L == 0) {
        uint l_scratch_cnt = atomic_load(&scratch_cnt);
        out_res[G] += l_scratch_cnt;
        printf("[final] G: %lu, scratch_cnt: %u\n", G, l_scratch_cnt);
    }
}

kernel void relaunch_kernel(__global start_condition* workspace, __global uint* workspace_sizes, __global ulong* out_res, unsigned state, unsigned recursion) {
    queue_t q = get_default_queue();
    
    printf("START RELAUNCH recursion: %u\n", recursion);

    uint limits[GPU_DEPTH - 1];
    // final run only depends on input limit, calculate differently
    limits[GPU_DEPTH-2] = workspace_sizes[GPU_DEPTH - 2];

    // Find maximum possible kernel launches
    //printf("state: 0x%x, recursion: %u\n", state, recursion);
    for(uint workspace_idx = 0; workspace_idx < (GPU_DEPTH - 2); workspace_idx++) {
        uint remaining = GPU_DEPTH - workspace_idx;
        uint input_limit = workspace_sizes[workspace_idx];
        uint output_limit = (WORKSPACE_SIZE - workspace_sizes[workspace_idx+1]) / remaining;
        printf("  size[%u] = %u, output_limit: %u\n",workspace_idx, workspace_sizes[workspace_idx], output_limit);
        limits[workspace_idx] = min(input_limit, output_limit);;
    }

    uint even_launches = 0;
    uint odd_launches = 0;
    printf("  size[%u] = %u\n", GPU_DEPTH-2, workspace_sizes[GPU_DEPTH-2]);

    for(uint workspace_idx = 0; workspace_idx < (GPU_DEPTH - 1); workspace_idx++) {
        if (workspace_idx % 2) {
            odd_launches += limits[workspace_idx];
        } else {
            even_launches += limits[workspace_idx];
        }
    }

    if (odd_launches == 0 && even_launches == 0) {
        printf("Finished, recursion: %u\n", recursion);
        return;
    }

#if 1
    // An unlimited recursion level potentially allows to fully use the GPU queue,
    // but this might be a corner case in the code and result in problems
    if (recursion > 500) {
        printf("Recursion limit\n");
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
            printf("Skipping, workspace_idx: %u\n", workspace_idx);
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
        if(launch_cnt > (WORK_FACTOR * WORKGROUP_SIZE)) {
            uint rest = launch_cnt % WORK_FACTOR;
            launch_cnt -= rest;
            applied_factor = WORK_FACTOR;
        }

        const __global start_condition *in_start = workspace + workspace_idx * WORKSPACE_SIZE + workspace_sizes[workspace_idx] - launch_cnt;
        __global start_condition *out_base = workspace + (workspace_idx + 1) * WORKSPACE_SIZE;
        __global uint *out_offs = workspace_sizes + workspace_idx + 1;

        void (^solve_2_blk)(void) = ^{
                    solve_final(in_start, out_res, applied_factor);
                };

#define SOLVE_BLK(remaining) void (^solve_##remaining##_blk)(void) = ^{solve_##remaining(in_start, out_base, out_offs, applied_factor);};

        SOLVE_BLK(3)
        SOLVE_BLK(4)
        SOLVE_BLK(5)
        SOLVE_BLK(6)
        SOLVE_BLK(7)
        SOLVE_BLK(8)
        SOLVE_BLK(9)
        SOLVE_BLK(10)
        
        void (^solve_single_blk)(void) = ^{
                    solve_single(in_start, out_base, out_offs, applied_factor);
                };
        
        uint remaining = GPU_DEPTH - workspace_idx;
        uint kernel_wg_size = 0;

#define BLK_WG_SIZE(remaining) case remaining: kernel_wg_size = get_kernel_work_group_size(solve_##remaining##_blk); break


        switch (remaining) {
            BLK_WG_SIZE(10);
            BLK_WG_SIZE(9);
            BLK_WG_SIZE(8);
            BLK_WG_SIZE(7);
            BLK_WG_SIZE(6);
            BLK_WG_SIZE(5);
            BLK_WG_SIZE(4);
            BLK_WG_SIZE(3);
            BLK_WG_SIZE(2);
        default:
            kernel_wg_size = get_kernel_work_group_size(solve_single_blk);
        }

        uint local_size = min((uint)WORKGROUP_SIZE, kernel_wg_size);
        local_size = min(local_size, launch_cnt/applied_factor);
        printf("local_size: %u, kernel_wg_size: %u\n", local_size, kernel_wg_size);
        printf("launch workspace_idx: %u, launch_cnt: %u, factor: %u, local_size: %u, utilization: %f\n", workspace_idx, launch_cnt, applied_factor, local_size, ((float)launch_cnt)/(float)WORKSPACE_SIZE);

#define ENQUEUE_BLK(remaining) case remaining: \
                                err = enqueue_kernel(q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, \
                                                     ndrange_1D(launch_cnt/applied_factor, local_size), \
                                                     0, \
                                                     NULL, \
                                                     &launched_kernels_evt[launched_kernels_cnt], \
                                                     solve_##remaining##_blk); break
        switch (remaining) {
            ENQUEUE_BLK(10);
            ENQUEUE_BLK(9);
            ENQUEUE_BLK(8);
            ENQUEUE_BLK(7);
            ENQUEUE_BLK(6);
            ENQUEUE_BLK(5);
            ENQUEUE_BLK(4);
            ENQUEUE_BLK(3);
            ENQUEUE_BLK(2);
        default:
            err = enqueue_kernel(q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
                                 ndrange_1D(launch_cnt/applied_factor, local_size),
                                 0,
                                 NULL,
                                 &launched_kernels_evt[launched_kernels_cnt],
                                 solve_single_blk);


        }

        if (err != 0) {
            printf("Error when enqueuing kernel, launch_cnt: %u, workspace_idx: %u, state: 0x%x, err: %d\n", launch_cnt, workspace_idx, state, err);
            goto cleanup_kernels_evt;
        } else  {
            printf("launch removed: %u\n", launch_cnt);
            // remove completed work items only when successfully launched
            workspace_sizes[workspace_idx] -= launch_cnt;
            launched_kernels_cnt++;
        }
    }

    printf("launched_kernels_cnt: %u\n", launched_kernels_cnt);

    clk_event_t marker_evt;
    err = enqueue_marker(q, launched_kernels_cnt, launched_kernels_evt, &marker_evt);
    if (err != 0) {
        printf("Error enqueuing marker\n");
        goto cleanup_kernels_evt;
    }

    void (^recursion_blk)(void) = ^{
                    relaunch_kernel(workspace, workspace_sizes, out_res, state, recursion + 1);
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
    
    printf("STOP  RELAUNCH recursion: %u\n", recursion);
}

kernel void sum_results(const __global ulong* res_in, __global ulong* res_out) {
    ulong cnt = 0;

    for(uint i = 0; i < SUM_REDUCTION_FACTOR; i++) {
        cnt += res_in[G*SUM_REDUCTION_FACTOR+i];
    }
    
    //printf("[sum  ] G: %lu, cnt: %lu\n", G, cnt);
    res_out[G] += cnt;
}
