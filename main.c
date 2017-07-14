#include <stdio.h>
#include <time.h>

#define MAXN 31
#define N 17

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
    int col_mask = ~(-1 << N); // mask with only the allowed columns set to 1
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

int nqueens(int n) {
  int q0, q1;

  int num = 0;
  int col_mask = -1 << n; // mask with only the allowed columns set to 1
  //
  // The top level is two fors, to save one bit of symmetry in the enumeration
  // by forcing second queen to
  // be AFTER the first queen.
  //
  for (q0 = 0; q0 < n - 2; q0++) {
    for (q1 = q0 + 2; q1 < n; q1++) {
      int bit0 = 1 << q0;
      bit_set[0] = bit0;
      int bit1 = 1 << q1;
      bit_set[1] = bit1;
      int d = 0; // d is our depth in the backtrack stack
      cols = bit0 | bit1 | (-1 << n); // The -1 here is used to fill all 'column' bits after n ...
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
        int clear_mask = bit_set[d] & ALIGN_DIA_L(diagl, d) & ALIGN_DIA_R(diagr, d);
        cols &= ~bit_set[d] | col_mask;
        diagl &= ~(ALIGN_COL_L(clear_mask, d));
        diagr &= ~(ALIGN_COL_R(clear_mask, d));

        candidates = posibs[d]; // backtrack ...
      }
    }
  }
  return num * 2;
}

main(int ac, char **av) {

  double time_diff, time_start; // for measuring calculation time
  time_start = clock();
  printf("Number of solution for %d is %d\n", N, nqueens(N));
  time_diff = (clock() - time_start); // calculating time difference
  printf("End Time:             %f seconds \n",
         time_diff / CLOCKS_PER_SEC); // needed time in seconds
}
