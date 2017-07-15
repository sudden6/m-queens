#include <stdio.h>
#include <time.h>
#include <stdint.h>

//#define TESTSUITE

#ifdef TESTSUITE
int n = 17;
#define N n
#define MAXN 31
#else
#define N 19
#define MAXN N
#endif

uint64_t nqueens() {
  int q0, q1;
  int cols[MAXN], diagl[MAXN], diagr[MAXN],
      posibs[MAXN]; // Our backtracking 'stack'
  uint64_t num = 0;
  //
  // The top level is two fors, to save one bit of symmetry in the enumeration
  // by forcing second queen to
  // be AFTER the first queen.
  //
  for (q0 = 0; q0 < N - 2; q0++) {
    for (q1 = q0 + 2; q1 < N; q1++) {
      int bit0 = 1 << q0;
      int bit1 = 1 << q1;
      int d = 0; // d is our depth in the backtrack stack
      cols[0] = bit0 | bit1 | (-1 << N); // The -1 here is used to fill all 'coloumn' bits after n ...
      diagl[0] = (1 << (2 + q0)) | (1 << (1 + q1));
      diagr[0] = (bit0 >> 2) | (bit1 >> 1);

      //  The variable posib contains the bitmask of possibilities we still have
      //  to try in a given row ...
      int posib = ~(cols[0] | diagl[0] | diagr[0]);

      while (d >= 0) {
          int diagl_shifted = diagl[d] << 1;
          int diagr_shifted = diagr[d] >> 1;
          while(posib) {
            int bit = posib & (~posib + 1); // The standard trick for getting the rightmost bit in the mask
            int ncols= cols[d] | bit;
            int ndiagl = (bit << 1) | diagl_shifted;
            int ndiagr = (bit >> 1) | diagr_shifted;
            int nposib = ~(ncols | ndiagl | ndiagr);
            posib^=bit; // Eliminate the tried possibility.

            if (nposib) {
                d += posib != 0;
              if (posib) { // This if saves stack depth + backtrack operations when we passed the last possibility in a row.
                //d++;
                posibs[d] = posib; // Go lower in stack ..
                //d++;
              }
              cols[d] = ncols;
              diagl[d] = ndiagl;
              diagr[d] = ndiagr;
              diagl_shifted = diagl[d] << 1;
              diagr_shifted = diagr[d] >> 1;
              posib = nposib;
            } else {
                num += ncols==-1;
            }
          }
          //d--;
          posib = posibs[d--]; // backtrack ...
          //d--;
      }
    }
  }
  return num * 2;
}

uint64_t results[19] = {1,     0,      0,       2,        10,      4,
                   40,    92,     352,     724,      2680,    14200,
                   73712, 365596, 2279184, 14772512, 95815104,
                   666090624,     4968057848};

int main(int argc, char **argv) {

#ifdef TESTSUITE
  int i;
  for (i = 1; i < 17; i++) {
    double time_diff, time_start; // for measuring calculation time
    n = i + 1;
    time_start = clock();
    int result = nqueens();
    time_diff = (clock() - time_start); // calculating time difference
    result == results[i] ? printf("PASS ") : printf("FAIL ");
    printf("N=%2d, Solutions=%10d, Expected=%10d, Time=%f s\n", n, result,
           results[i], time_diff / CLOCKS_PER_SEC);
  }
#else
  double time_diff, time_start; // for measuring calculation time
  time_start = clock();
  int result = nqueens();
  time_diff = (clock() - time_start); // calculating time difference
  result == results[N - 1] ? printf("PASS ") : printf("FAIL ");
  printf("N=%2d, Solutions=%10d, Expected=%10d, Time=%f s\n", N, result,
         results[N - 1], time_diff / CLOCKS_PER_SEC);
#endif
  return 0;
}
