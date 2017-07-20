#include <stdint.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>

//#define TESTSUITE

#ifdef TESTSUITE
int n = 17;
#define N n
#define MAXN 31
#else
#define N 17
#define MAXN N
#endif

// get the current wall clock time in seconds
double get_time() {
  struct timeval tp;
  struct timezone tz;
  gettimeofday(&tp, &tz);
  return tp.tv_sec + tp.tv_usec / 1000000.0;
}

uint64_t nqueens() {

  // counter for the number of solutions
  // sufficient until n=29
  int num = 0;
//
// The top level is two fors, to save one bit of symmetry in the enumeration
// by forcing second queen to be AFTER the first queen.
//

#pragma omp parallel for reduction(+ : num) schedule(dynamic)
  for (int q0 = 0; q0 < N - 2; q0++) {
    for (int q1 = q0 + 2; q1 < N; q1++) {
      uint_fast32_t cols[MAXN], posibs[MAXN];   // Our backtracking 'stack'
      uint64_t diagl[MAXN], diagr[MAXN];
      uint_fast32_t bit0 = 1 << q0; // The first queen placed
      uint_fast32_t bit1 = 1 << q1; // The second queen placed
      int d = 0;          // d is our depth in the backtrack stack
      // The -1 here is used to fill all 'coloumn' bits after n ...
      cols[0] = bit0 | bit1 | (-1 << N);
      // The next two lines are done with different algorithms, this somehow
      // improves performance a bit...
      diagl[0] = (1 << (2 + q0)) | (1 << (1 + q1));
      diagr[0] = (bit0 >> 2) | (bit1 >> 1);

      //  The variable posib contains the bitmask of possibilities we still have
      //  to try in a given row ...
      uint_fast32_t posib = ~(cols[0] | diagl[0] | diagr[0]);

      while (d >= 0) {
        // moving the two shifts out of the inner loop slightly improves
        // performance
        uint64_t diagl_shifted = diagl[d] << 1;
        uint64_t diagr_shifted = diagr[d] >> 1;
        while (posib) {
          uint_fast32_t bit = posib & (~posib + 1); // The standard trick for getting the
                                          // rightmost bit in the mask
          uint_fast32_t new_cols = cols[d] | bit;
          uint64_t new_diagl = (bit << 1) | diagl_shifted;
          uint64_t new_diagr = (bit >> 1) | diagr_shifted;
          uint_fast32_t new_posib = ~(new_cols | new_diagl | new_diagr);
          posib ^= bit; // Eliminate the tried possibility.

          if (new_posib) {
            // increment d if there are possibilities left,
            // not doing this in the if yields better performance
            d += posib != 0;
            if (posib) { // This if saves stack depth + backtrack operations
                         // when we passed the last possibility in a row.
              posibs[d] = posib; // Go lower in stack ..
            }

            // make values current
            posib = new_posib;
            cols[d] = new_cols;
            diagl[d] = new_diagl;
            diagr[d] = new_diagr;

            diagl_shifted = new_diagl << 1;
            diagr_shifted = new_diagr >> 1;
          } else {
            // when all columns are used, we found a solution
            num += new_cols == -1;
          }
        }
        posib = posibs[d--]; // backtrack ...
      }
    }
  }
  return num * 2;
}

uint64_t results[19] = {1,        0,        0,         2,         10,
                        4,        40,       92,        352,       724,
                        2680,     14200,    73712,     365596,    2279184,
                        14772512, 95815104, 666090624, 4968057848};

int main(int argc, char **argv) {

#ifdef TESTSUITE
  int i;
  for (i = 1; i < 17; i++) {
    double time_diff, time_start; // for measuring calculation time
    n = i + 1;
    time_start = get_time();
    uint64_t result = nqueens();
    time_diff = (get_time() - time_start); // calculating time difference
    result == results[i] ? printf("PASS ") : printf("FAIL ");
    printf("N=%2d, Solutions=%10d, Expected=%10d, Time=%fs, Solutions/s= %f\n",
           n, result, results[i], time_diff, result / time_diff);
  }
#else
  double time_diff, time_start; // for measuring calculation time
  time_start = get_time();
  uint64_t result = nqueens();
  time_diff = (get_time() - time_start); // calculating time difference
  result == results[N - 1] ? printf("PASS ") : printf("FAIL ");
  printf("N=%2d, Solutions=%10d, Expected=%10d, Time=%fs, Solutions/s= %f\n", N,
         result, results[N - 1], time_diff, result / time_diff);
#endif
  return 0;
}
