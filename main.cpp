#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#include <vector>

// uncomment to start with n=2 and compare to known results
#define TESTSUITE

#ifndef N
#define N 12
#endif
#define MAXN 29

#if N > MAXN
#warning "N too big, overflow may occur"
#endif

#if N < 2
#error "N too small"
#endif

// get the current wall clock time in seconds
double get_time() {
  struct timeval tp;
  gettimeofday(&tp, nullptr);
  return tp.tv_sec + tp.tv_usec / 1000000.0;
}

typedef struct {
    uint_fast32_t cols; // bitfield with all the used columns
    uint_fast32_t diagl;// bitfield with all the used diagonals down left
    uint_fast32_t diagr;// bitfield with all the used diagonals down right
    int_fast32_t placed;// number of rows where queens are already placed
} start_condition;

std::vector<start_condition> create_subboards_s1(uint_fast8_t n) {
    std::vector<start_condition> result;

    if(n < 2) {
        return  result;
    }

    //
    // The top level is two fors, to save one bit of symmetry in the enumeration
    // by forcing second queen to be AFTER the first queen.
    //
    // maximum size needed for storing all results
    uint_fast16_t num_starts = (n-2)*(n-1);
    uint_fast16_t start_cnt = 0;
    result.resize(num_starts);         // preallocate memory
    #pragma omp simd
    for (uint_fast8_t q0 = 0; q0 < n - 2; q0++) {
      for (uint_fast8_t q1 = q0 + 2; q1 < n; q1++) {
        uint_fast32_t bit0 = 1 << q0;
        uint_fast32_t bit1 = 1 << q1;
        result[start_cnt].cols = bit0 | bit1;
        result[start_cnt].diagl = (bit0 << 2) | (bit1 << 1);
        result[start_cnt].diagr = (bit0 >> 2) | (bit1 >> 1);
        result[start_cnt].placed = 2;
        start_cnt++;
      }
    }
    result.resize(start_cnt); // shrink
}

uint64_t solve_subboard(uint_fast8_t n, std::vector<start_condition> starts) {

  // counter for the number of solutions
  // sufficient until n=29
  uint_fast64_t num = 0;
  size_t start_cnt = starts.size();

#pragma omp parallel for reduction(+ : num) schedule(dynamic)
  for (uint_fast16_t cnt = 0; cnt < start_cnt; cnt++) {
    uint_fast32_t cols[MAXN], posibs[MAXN]; // Our backtracking 'stack'
    uint_fast32_t diagl[MAXN], diagr[MAXN];
    int_fast32_t rest[MAXN]; // number of rows left
    int_fast16_t d = 1; // d is our depth in the backtrack stack
    // The UINT_FAST32_MAX here is used to fill all 'coloumn' bits after n ...
    cols[d] = starts[cnt].cols | (UINT_FAST32_MAX << n);
    // This places the first two queens
    diagl[d] = starts[cnt].diagl;
    diagr[d] = starts[cnt].diagr;
#define LOOKAHEAD 3
    // we're allready two rows into the field here
    rest[d] = n - LOOKAHEAD - starts[start_cnt].placed;

    //  The variable posib contains the bitmask of possibilities we still have
    //  to try in a given row ...
    uint_fast32_t posib = (cols[d] | diagl[d] | diagr[d]);

    while (d > 0) {
      // moving the two shifts out of the inner loop slightly improves
      // performance
      uint_fast32_t diagl_shifted = diagl[d] << 1;
      uint_fast32_t diagr_shifted = diagr[d] >> 1;
      int_fast32_t l_rest = rest[d];

      while (posib != UINT_FAST32_MAX) {
        // The standard trick for getting the rightmost bit in the mask
        uint_fast32_t bit = ~posib & (posib + 1);
        uint_fast32_t new_diagl = (bit << 1) | diagl_shifted;
        uint_fast32_t new_diagr = (bit >> 1) | diagr_shifted;
        uint_fast32_t new_posib = (cols[d] | bit | new_diagl | new_diagr);
        posib ^= bit; // Eliminate the tried possibility.
        bit |= cols[d];

        if (new_posib != UINT_FAST32_MAX) {
            uint_fast32_t lookahead = (bit | (new_diagl << (LOOKAHEAD - 2)) | (new_diagr >> (LOOKAHEAD - 2)));
            uint_fast32_t lookahead2 = (bit | (new_diagl << (LOOKAHEAD - 1)) | (new_diagr >> (LOOKAHEAD - 1)));
            uint_fast32_t allowed1 = l_rest >= 0;
            uint_fast32_t allowed2 = l_rest > 0;

            if(allowed1 && (lookahead == UINT_FAST32_MAX)) {
                continue;
            }

            if(allowed2 && (lookahead2 == UINT_FAST32_MAX)) {
                continue;
            }

          // The next two lines save stack depth + backtrack operations
          // when we passed the last possibility in a row.
          // Go lower in the stack, avoid branching by writing above the current
          // position
          posibs[d + 1] = posib;
          d += posib != UINT_FAST32_MAX; // avoid branching with this trick
          posib = new_posib;

          l_rest--;

          // make values current
          cols[d] = bit;
          diagl[d] = new_diagl;
          diagr[d] = new_diagr;
          rest[d] = l_rest;
          diagl_shifted = new_diagl << 1;
          diagr_shifted = new_diagr >> 1;
        } else {
          // when all columns are used, we found a solution
          num += bit == UINT_FAST32_MAX;
        }
      }
      posib = posibs[d]; // backtrack ...
      d--;
    }
  }
  return num * 2;
}

// expected results from https://oeis.org/A000170
uint64_t results[27] = {1ULL,
                        0ULL,
                        0ULL,
                        2ULL,
                        10ULL,
                        4ULL,
                        40ULL,
                        92ULL,
                        352ULL,
                        724ULL,
                        2680ULL,
                        14200ULL,
                        73712ULL,
                        365596ULL,
                        2279184ULL,
                        14772512ULL,
                        95815104ULL,
                        666090624ULL,
                        4968057848ULL,
                        39029188884ULL,
                        314666222712ULL,
                        2691008701644ULL,
                        24233937684440ULL,
                        227514171973736ULL,
                        2207893435808352ULL,
                        22317699616364044ULL,
                        234907967154122528ULL};

int main(int argc, char **argv) {

#ifdef TESTSUITE
  int i = 2;
#else
  int i = N;
#endif
  if (argc == 2) {
    i = atoi(argv[1]);
    if (i < 1 || i > MAXN) {
      printf("n must be between 2 and %d!\n", MAXN);
    }
  }

  for (; i <= N; i++) {
    double time_diff, time_start; // for measuring calculation time
    time_start = get_time();
    std::vector<start_condition> st = create_subboards_s1(i);
    uint64_t result = solve_subboard(i, st);
    time_diff = (get_time() - time_start); // calculating time difference
    result == results[i - 1] ? printf("PASS ") : printf("FAIL ");
    printf("N %2d, Solutions %18" PRIu64 ", Expected %18" PRIu64
           ", Time %fs, Solutions/s %f\n",
           i, result, results[i - 1], time_diff, result / time_diff);
  }
  return 0;
}
