#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

// uncomment to start with n=2 and compare to known results
//#define TESTSUITE

#ifndef N
#define N 17
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
  gettimeofday(&tp, NULL);
  return tp.tv_sec + tp.tv_usec / 1000000.0;
}

uint64_t nqueens(uint_fast8_t n) {

  // counter for the number of solutions
  // sufficient until n=29
  uint_fast64_t num = 0;
  //
  // The top level is two fors, to save one bit of symmetry in the enumeration
  // by forcing second queen to be AFTER the first queen.
  //
  uint_fast16_t num_starts = ((n - 2) * (n - 2) * (n - 1)) / 2;
  uint_fast16_t start_cnt = 0;
  uint_fast32_t start_queens[num_starts][2];
  #pragma omp simd
  for (uint_fast8_t q0 = 0; q0 < n - 2; q0++) {
    for (uint_fast8_t q1 = q0 + 2; q1 < n; q1++) {
      start_queens[start_cnt][0] = 1 << q0;
      start_queens[start_cnt][1] = 1 << q1;
      start_cnt++;
    }
  }

//#pragma omp parallel for reduction(+ : num) schedule(dynamic)
  for (uint_fast16_t cnt = 0; cnt < start_cnt; cnt++) {
    uint_fast32_t cols[MAXN], posibs[MAXN]; // Our backtracking 'stack'
    uint_fast32_t diagl[MAXN], diagr[MAXN];
    uint_fast32_t bit0 = start_queens[cnt][0]; // The first queen placed
    uint_fast32_t bit1 = start_queens[cnt][1]; // The second queen placed
    int_fast16_t d = 1; // d is our depth in the backtrack stack
    // The UINT_FAST32_MAX here is used to fill all 'coloumn' bits after n ...
    cols[d] = bit0 | bit1 | (UINT_FAST32_MAX << n);
    // This places the first two queens
    diagl[d] = (bit0 << 2) | (bit1 << 1);
    diagr[d] = (bit0 >> 2) | (bit1 >> 1);

    //  The variable posib contains the bitmask of possibilities we still have
    //  to try in a given row ...
    uint_fast32_t posib = (cols[d] | diagl[d] | diagr[d]);

    while (d > 0) {
      // moving the two shifts out of the inner loop slightly improves
      // performance
      uint_fast32_t diagl_shifted = diagl[d] << 1;
      uint_fast32_t diagr_shifted = diagr[d] >> 1;
      while (posib != UINT_FAST32_MAX) {
        // The standard trick for getting the rightmost bit in the mask
        uint_fast32_t bit = ~posib & (posib + 1);
        uint_fast32_t new_cols = cols[d] | bit;
        uint_fast32_t new_diagl = (bit << 1) | diagl_shifted;
        uint_fast32_t new_diagr = (bit >> 1) | diagr_shifted;
        uint_fast32_t new_posib = (new_cols | new_diagl | new_diagr);
        posib ^= bit; // Eliminate the tried possibility.

        if (new_posib != UINT_FAST32_MAX) {
          // The next two lines save stack depth + backtrack operations
          // when we passed the last possibility in a row.
          // Go lower in the stack, avoid branching by writing above the current
          // position
          posibs[d + 1] = posib;
          d += posib != UINT_FAST32_MAX; // avoid branching with this trick


          // make values current
          posib = new_posib;
          cols[d] = new_cols;
          diagl[d] = new_diagl;
          diagr[d] = new_diagr;

          diagl_shifted = new_diagl << 1;
          diagr_shifted = new_diagr >> 1;
        } else {
          // when all columns are used, we found a solution
          num += new_cols == UINT_FAST32_MAX;
        }
      }
      posib = posibs[d--]; // backtrack ...
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
    uint64_t result = nqueens(i);
    time_diff = (get_time() - time_start); // calculating time difference
    result == results[i - 1] ? printf("PASS ") : printf("FAIL ");
    printf("N %2d, Solutions %18" PRIu64 ", Expected %18" PRIu64
           ", Time %fs, Solutions/s %f\n",
           i, result, results[i - 1], time_diff, result / time_diff);
  }
  return 0;
}
