#include <stdio.h>
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

// align diag values to columns
#define ALIGN_DIA_R(x, d) ((x) >> (d))
#define ALIGN_DIA_L(x, d) ((x) >> (N - 1 - (d)))

// align column value to diag
#define ALIGN_COL_R(x, d) ((x) << (d))
#define ALIGN_COL_L(x, d) ((x) << (N - 1 - (d)))

int cols, diagl[MAXN], diagr[MAXN];
int bit_set[MAXN] = {0};


void printChessBoard()
{
    int row;
    int col;
    printf("\n");

    for(row = 0; row  < N; row++) {
        for(col = 0; col < N; col++) {
            if(bit_set[row] & (1 << (N - 1 - col))) {
                printf("X ");		// queen
            } else {
                printf("- ");		// empty field
            }
        }
        printf("\n");
    }
}

int nqueens() {
  int q0, q1;

  int num = 0;
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
      cols = bit0 | bit1 | (-1 << N); // The -1 here is used to fill all 'column' bits after n ...
      diagl[0]= (bit0<<1 | bit1)<<1;
      diagr[0]= (bit0>>1 | bit1)>>1;

      //  The variable candidates contains the bitmask of possibilities we still
      //  have to try in a given row ...
      int candidates = ~(cols | diagl[0] | diagr[0]);

      while (d >= 0) {
          int bit = 0;
        while (candidates) {
          bit = candidates & -candidates; // The standard trick for getting
                                              // the rightmost bit in the mask
          int ncols = cols | bit;
          int nxt_diagl = (diagl[d] | bit) << 1;
          int nxt_diagr = (diagr[d] | bit) >> 1;
          int nxt_possible = ~(ncols | nxt_diagl | nxt_diagr);

          num += ncols == -1;

          if (nxt_possible) {
            d++;
            bit_set[d] = candidates;
            cols = ncols;
            diagl[d] = nxt_diagl;
            diagr[d] = nxt_diagr;
            candidates = nxt_possible;
          } else {
              candidates ^= bit; // Eliminate the tried possibility.
          }
        }
            int last_set = bit_set[d] & -bit_set[d];
            cols &= ~last_set;
            candidates = ~last_set & bit_set[d];
            d--;
      }
    }
  }
  return num * 2;
}

int results[17] = {1, 0, 0, 2, 10, 4, 40, 92, 352,
                   724, 2680, 14200, 73712, 365596,
                   2279184, 14772512, 95815104};

int main(int argc, char argv) {

#ifdef TESTSUITE
    int i;
    for(i = 1; i < 17; i++) {
        double time_diff, time_start; // for measuring calculation time
        n = i + 1;
        time_start = clock();
        int result = nqueens();
        time_diff = (clock() - time_start); // calculating time difference
        result == results[i] ? printf("PASS ") : printf("FAIL ");
        printf("N=%2d, Solutions=%10d, Expected=%10d, Time=%f s\n",n , result, results[i], time_diff / CLOCKS_PER_SEC);
    }
#else
    double time_diff, time_start; // for measuring calculation time
    time_start = clock();
    int result = nqueens();
    time_diff = (clock() - time_start); // calculating time difference
    result == results[N - 1] ? printf("PASS ") : printf("FAIL ");
    printf("N=%2d, Solutions=%10d, Expected=%10d, Time=%f s\n", N , result, results[N-1], time_diff / CLOCKS_PER_SEC);
#endif
    return 0;
}
