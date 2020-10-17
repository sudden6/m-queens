typedef struct __attribute__ ((packed)) {
    uint cols; // bitfield with all the used columns
    uint diagl;// bitfield with all the used diagonals down left
    uint diagr;// bitfield with all the used diagonals down right
} start_condition;

#define MAXN 29

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
#define WORKSPACE_SIZE 16
//#define GPU_DEPTH 2
//#define BOARDSIZE 8

#define LAST_PLACED (BOARDSIZE - 1)
#define FIRST_PLACED (BOARDSIZE - GPU_DEPTH)

uint get_tmp_idx(uint placed) {
    return placed - FIRST_PLACED;
}

kernel void relaunch_kernel(__global start_condition* workspace, __global uint* workspace_sizes, __global uint* out_res, unsigned placed);

// without lookahead optimization
kernel void solve_single_no_look(__global start_condition* workspace, __global uint* workspace_sizes, __global uint* out_res, unsigned placed) {

    // input index calculation
    uint in_first_idx = get_tmp_idx(placed) * WORKSPACE_SIZE;
    uint in_our_idx = in_first_idx + G;

    // output index calculation
    uint out_first_idx = get_tmp_idx(placed + 1) * WORKSPACE_SIZE;
    __global uint *out_cur_idx_ptr = workspace_sizes + get_tmp_idx(placed + 1);

    // algorithm start
    uint_fast32_t cols = workspace[in_our_idx].cols;
    uint_fast32_t diagl = workspace[in_our_idx].diagl;
    uint_fast32_t diagr = workspace[in_our_idx].diagr;

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
            uint out_offs = atomic_add(out_cur_idx_ptr, 1);
            uint out_idx = out_first_idx + out_offs;

            workspace[out_idx].cols = bit;
            workspace[out_idx].diagr = new_diagl;
            workspace[out_idx].diagl = new_diagl;
        }
    }

    // launch relaunch_kernel
    if(G == 0) {
        queue_t q = get_default_queue();
        enqueue_kernel(q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
                      ndrange_1D(1),
                      ^{
                           relaunch_kernel(workspace, workspace_sizes, out_res, placed);
                       });

    }
}

kernel void solve_final(__global start_condition* workspace, __global uint* workspace_sizes, __global uint* out_res) {
    // input index calculation
    uint in_first_idx = get_tmp_idx(LAST_PLACED) * WORKSPACE_SIZE;
    uint in_our_idx = in_first_idx + G;

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

    while (posib != UINT_FAST32_MAX) {
        // The standard trick for getting the rightmost bit in the mask
        uint_fast32_t bit = ~posib & (posib + 1);
        posib ^= bit; // Eliminate the tried possibility.
        uint_fast32_t new_diagl = (bit << 1) | diagl;
        uint_fast32_t new_diagr = (bit >> 1) | diagr;
        bit |= cols;
        uint_fast32_t new_posib = (bit | new_diagl | new_diagr);

        if (new_posib != UINT_FAST32_MAX) {
            cnt++;
        }
    }

    out_res += cnt;
    // launch relaunch_kernel
    if(G == 0) {
        queue_t q = get_default_queue();
        enqueue_kernel(q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
                      ndrange_1D(1),
                      ^{
                           relaunch_kernel(workspace, workspace_sizes, out_res, FIRST_PLACED);
                       });

    }
}



kernel void relaunch_kernel(__global start_condition* workspace, __global uint* workspace_sizes, __global uint* out_res, unsigned placed) {
    queue_t q = get_default_queue();
    uint next_placed = placed + 1;
    // limit due to input start conditions
    uint max_launches = workspace_sizes[get_tmp_idx(placed)];

    while(placed < LAST_PLACED) {
        // single solver step

        // compute maximum safe launches due to data expansion in output start conditions
        uint output_limit = (WORKSPACE_SIZE - workspace_sizes[get_tmp_idx(next_placed)]) / expansion_factor(next_placed);
        max_launches = min(max_launches, output_limit);

        if(max_launches == 0) {
            // can't launch here, go deeper
            placed++;
            next_placed = placed + 1;
            max_launches = workspace_sizes[get_tmp_idx(placed)];
            continue;
        }

        // launch kernel
        enqueue_kernel(q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
                      ndrange_1D(max_launches),
                      ^{
                           solve_single_no_look(workspace, workspace_sizes, out_res, next_placed);
                       });

    }

    if(placed == LAST_PLACED) {
        // last step, count solutions

        // launch kernel
        enqueue_kernel(q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
                      ndrange_1D(max_launches),
                      ^{
                           solve_final(workspace, workspace_sizes, out_res);
                       });
    }

    // remove completed work items
    workspace_sizes[get_tmp_idx(placed)] -= max_launches;
}
