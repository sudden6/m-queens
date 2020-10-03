typedef struct __attribute__ ((packed)) {
    uint diagr;// bitfield with all the used diagonals down left
    uint diagl;// bitfield with all the used diagonals down right
} diags_packed_t;

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

kernel void count_solutions(__global const diags_packed_t* lut, __global const diags_packed_t* candidates, __global uint* out_cnt, uint lut_offset) {
    uint cnt = 0;
    uint lut_diagr = lut[G + lut_offset].diagr;
    uint lut_diagl = lut[G + lut_offset].diagl;

    for(int i = 0; i < MAX_CANDIDATES; i++) {
        cnt += ((lut_diagr & candidates[i].diagr) == 0) && ((lut_diagl & candidates[i].diagl) == 0);
    }

    //printf("G: %d, L: %d, cnt: %d, lut_diagr: %x, lut_diagl: %x\n", G, L, cnt, lut_diagr, lut_diagl);

    out_cnt[G] += cnt;
}

kernel void count_solutions_trans(__global const diags_packed_t* lut, __global const diags_packed_t* candidates, __global uint* out_cnt, uint lut_offset, uint lut_count) {
    uint cnt = 0;

    for(int i = 0; i < lut_count; i++) {
        uint lut_diagr = lut[i + lut_offset].diagr;
        uint lut_diagl = lut[i + lut_offset].diagl;
        cnt += ((lut_diagr & candidates[G].diagr) == 0) && ((lut_diagl & candidates[G].diagl) == 0);
    }

    //printf("G: %d, L: %d, cnt: %d, lut_diagr: %x, lut_diagl: %x\n", G, L, cnt, lut_diagr, lut_diagl);

    out_cnt[G] += cnt;
}
