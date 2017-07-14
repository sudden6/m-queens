#include <stdio.h>
#include <time.h>

#define MAXN 31
#define N 5

// align diag values to columns
#define ALIGN_DIA_R(x, d) ((x) >> (d))
#define ALIGN_DIA_L(x, d) ((x) >> (N - 1 - (d)))

// align column value to diag
#define ALIGN_COL_R(x, d) ((x) << (d))
#define ALIGN_COL_L(x, d) ((x) << (N - 1 - (d)))

int nqueens(int n) {
  int q0, q1;
  int cols, diagl, diagr;
  int posibs[MAXN]; // Our backtracking 'stack'
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
      int bit1 = 1 << q1;
      int d = 0; // d is our depth in the backtrack stack
      cols = bit0 | bit1 | (-1 << n); // The -1 here is used to fill all 'column' bits after n ...
      diagl = ALIGN_COL_L(bit0, 0) | ALIGN_COL_L(bit1, 1);
      diagr = ALIGN_COL_R(bit0, 0) | ALIGN_COL_R(bit1, 1);
      d = 2; // two queens already placed

      //  The variable candidates contains the bitmask of possibilities we still
      //  have to try in a given row ...
      int candidates = ~(cols | ALIGN_DIA_L(diagl, d) | ALIGN_DIA_R(diagr, d));

      while (d >= 2) {
        while (candidates) {
          int bit = candidates & -candidates; // The standard trick for getting
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

          if (nxt_possible) {
            if (candidates) { // This if saves stack depth + backtrack
                              // operations when we passed the last possibility
                              // in a row.
              d++;
              posibs[d] = candidates; // Go lower in stack ..
            }
            cols = ncols;
            diagl = nxt_diagl;
            diagr = nxt_diagr;
            candidates = nxt_possible;
          } else {
              int clear_mask = cols & ALIGN_DIA_L(diagl, d) & ALIGN_DIA_R(diagr, d);
              cols &= ~clear_mask | col_mask;
              diagl &= ~(ALIGN_COL_L(clear_mask, d));
              diagr &= ~(ALIGN_COL_R(clear_mask, d));
          }
        }
        //TODO: check if col_mask needed
        d--;


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
