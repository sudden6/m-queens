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

int cols, diagl, diagr;
int posibs[MAXN] = {0}; // Our backtracking 'stack'
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
      diagl = ALIGN_COL_L(bit0, 0) | ALIGN_COL_L(bit1, 1);
      diagr = ALIGN_COL_R(bit0, 0) | ALIGN_COL_R(bit1, 1);
      d = 2; // two queens already placed

      //  The variable candidates contains the bitmask of possibilities we still
      //  have to try in a given row ...
      int candidates = ~(cols | ALIGN_DIA_L(diagl, d) | ALIGN_DIA_R(diagr, d));

      while (d >= 2) {
          int bit = 0;
        while (candidates) {
          bit = candidates & -candidates; // The standard trick for getting
                                              // the rightmost bit in the mask
          int ncols = cols | bit;
          int nxt_diagl = ALIGN_COL_L(bit, d) | diagl;
          int nxt_diagr = ALIGN_COL_R(bit, d) | diagr;
          int nxt_possible = ~(ncols | ALIGN_DIA_L(nxt_diagl, d + 1) | ALIGN_DIA_R(nxt_diagr, d + 1));
          candidates ^= bit; // Eliminate the tried possibility.

          // The following is the main additional trick here, as recognizing
          // solution can not be done using stack level (d),
          // since we save the depth+backtrack time at the end of the
          // enumeration loop. However by noticing all coloumns are
          // filled (comparison to -1) we know a solution was reached ...
          // Notice also that avoiding an if on the ncols==-1 comparison is more
          // efficient!
          num += ncols == -1;

          /*
          if(ncols == -1) {
              num++;
              printChessBoard();
          }//*/

          if (nxt_possible) {
            posibs[d] = candidates;
            bit_set[d] = bit;
            d++;
            cols = ncols;
            diagl = nxt_diagl;
            diagr = nxt_diagr;
            candidates = nxt_possible;
          }
        }
        d--;
        cols &= ~bit_set[d];
        diagl &= ~(ALIGN_COL_L(bit_set[d], d));
        diagr &= ~(ALIGN_COL_R(bit_set[d], d));

        candidates = posibs[d]; // backtrack ...
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
