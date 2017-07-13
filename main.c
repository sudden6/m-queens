#include <stdio.h>
#include <time.h>

#define MAXN 31
#define N 17

int nqueens(int n)
{
  int q0,q1;
  int cols[MAXN], diagl[MAXN], diagr[MAXN], posibs[MAXN]; // Our backtracking 'stack'
  int num=0;
  //
  // The top level is two fors, to save one bit of symmetry in the enumeration by forcing second queen to
  // be AFTER the first queen.
  //
  for (q0=0; q0<n-2; q0++) {
    for (q1=q0+2; q1<n; q1++){
      int bit0 = 1<<q0;
      int bit1 = 1<<q1;
      int d=0; // d is our depth in the backtrack stack
      cols[0] = bit0 | bit1 | (-1<<n); // The -1 here is used to fill all 'coloumn' bits after n ...
      diagl[0]= (bit0<<1 | bit1)<<1;
      diagr[0]= (bit0>>1 | bit1)>>1;

      //  The variable candidates contains the bitmask of possibilities we still have to try in a given row ...
      int candidates = ~(cols[0] | diagl[0] | diagr[0]);

      while (d >= 0) {
        while(candidates) {
          int bit = candidates & -candidates; // The standard trick for getting the rightmost bit in the mask
          int ncols= cols[d] | bit;
          int ndiagl = (diagl[d] | bit) << 1;
          int ndiagr = (diagr[d] | bit) >> 1;
          int nposib = ~(ncols | ndiagl | ndiagr);
          candidates^=bit; // Eliminate the tried possibility.

          // The following is the main additional trick here, as recognizing solution can not be done using stack level (d),
          // since we save the depth+backtrack time at the end of the enumeration loop. However by noticing all coloumns are
          // filled (comparison to -1) we know a solution was reached ...
          // Notice also that avoiding an if on the ncols==-1 comparison is more efficient!
          num += ncols==-1;

          if (nposib) {
            if (candidates) { // This if saves stack depth + backtrack operations when we passed the last possibility in a row.
              posibs[d++] = candidates; // Go lower in stack ..
            }
            cols[d] = ncols;
            diagl[d] = ndiagl;
            diagr[d] = ndiagr;
            candidates = nposib;
          }
        }
        candidates = posibs[--d]; // backtrack ...
      }
    }
  }
  return num*2;
}


main(int ac , char **av)
{

  double time_diff, time_start;   // for measuring calculation time
  time_start = clock();
  printf("Number of solution for %d is %d\n",N,nqueens(N));
  time_diff = (clock() - time_start); // calculating time difference
  printf ("End Time:             %f seconds \n", time_diff/CLOCKS_PER_SEC);  // needed time in seconds
}
