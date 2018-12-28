#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

// above this board size overflows may occur
#define MAXN 29

// get the current wall clock time in seconds
double get_time() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return tp.tv_sec + tp.tv_usec / 1000000.0;
}

uint64_t nqueens(uint_fast8_t n) {

  // counter for the number of solutions
  // sufficient until n=29 (estimated)
  uint_fast64_t num = 0;
  //
  // The top level is two fors, to save one bit of symmetry in the enumeration
  // by forcing second queen to be AFTER the first queen.
  //
  uint_fast16_t start_cnt = 0;
  uint_fast32_t start_queens[(MAXN - 2)*(MAXN - 1)][2];
  #pragma omp simd
  for (uint_fast8_t q0 = 0; q0 < n - 2; q0++) {
    for (uint_fast8_t q1 = q0 + 2; q1 < n; q1++) {
      start_queens[start_cnt][0] = 1 << q0;
      start_queens[start_cnt][1] = 1 << q1;
      start_cnt++;
    }
  }

  // maximum number of queens present at row 3
  #define START_LEVEL3 ((MAXN - 3) * (MAXN - 2) * (MAXN - 1))
  // store the state at level3 as cols, diagl, diagr
  uint_fast32_t start_level3[START_LEVEL3][3];

    uint_fast32_t start_level3_cnt = 0;
#define START_LEVEL4 (MAXN - 4)

#pragma omp parallel for schedule(dynamic)
  for (uint_fast16_t cnt = 0; cnt < start_cnt; cnt++) {
    uint_fast32_t start_level4[START_LEVEL4][3];
    uint_fast32_t level4_cnt = 0;
    uint_fast32_t cols[MAXN], posibs[MAXN]; // Our backtracking 'stack'
    uint_fast32_t diagl[MAXN], diagr[MAXN];
    int8_t rest[MAXN]; // number of rows left
    uint_fast32_t bit0 = start_queens[cnt][0]; // The first queen placed
    uint_fast32_t bit1 = start_queens[cnt][1]; // The second queen placed
    int_fast16_t d = 1; // d is our depth in the backtrack stack
    // The UINT_FAST32_MAX here is used to fill all 'coloumn' bits after n ...
    cols[d] = bit0 | bit1 | (UINT_FAST32_MAX << n);
    // This places the first two queens
    diagl[d] = (bit0 << 2) | (bit1 << 1);
    diagr[d] = (bit0 >> 2) | (bit1 >> 1);
#define LOOKAHEAD 3
    // we're allready two rows into the field here
    rest[d] = n - 2 - LOOKAHEAD;

#define STORE_LEVEL (n - 2 - LOOKAHEAD - 1)

    //  The variable posib contains the bitmask of possibilities we still have
    //  to try in a given row ...
    uint_fast32_t posib = (cols[d] | diagl[d] | diagr[d]);

    diagl[d] <<= 1;
    diagr[d] >>= 1;

    while (d > 0) {
      int8_t l_rest = rest[d];

      while (posib != UINT_FAST32_MAX) {
        // The standard trick for getting the rightmost bit in the mask
        uint_fast32_t bit = ~posib & (posib + 1);
        posib ^= bit; // Eliminate the tried possibility.
        uint_fast32_t new_diagl = (bit << 1) | diagl[d];
        uint_fast32_t new_diagr = (bit >> 1) | diagr[d];
        bit |= cols[d];
        uint_fast32_t new_posib = (bit | new_diagl | new_diagr);

        if (new_posib != UINT_FAST32_MAX) {
            uint_fast32_t lookahead1 = (bit | (new_diagl << (LOOKAHEAD - 2)) | (new_diagr >> (LOOKAHEAD - 2)));
            uint_fast32_t lookahead2 = (bit | (new_diagl << (LOOKAHEAD - 1)) | (new_diagr >> (LOOKAHEAD - 1)));
            uint_fast32_t allowed2 = l_rest > (int8_t)0;

            if(allowed2 && ((lookahead2 == UINT_FAST32_MAX) || (lookahead1 == UINT_FAST32_MAX))) {
                continue;
            }


          if(l_rest == (STORE_LEVEL + 1)) {
            start_level4[level4_cnt][0] = bit;   // cols
            start_level4[level4_cnt][1] = new_diagl; // diagl
            start_level4[level4_cnt][2] = new_diagr; // diagr
            level4_cnt++;
            continue;
          }

          l_rest--;

          // The next two lines save stack depth + backtrack operations
          // when we passed the last possibility in a row.
          // Go lower in the stack, avoid branching by writing above the current
          // position
          posibs[d] = posib;
          d += posib != UINT_FAST32_MAX; // avoid branching with this trick
          posib = new_posib;


          // make values current
          cols[d] = bit;
          diagl[d] = new_diagl << 1;
          diagr[d] = new_diagr >> 1;
          rest[d] = l_rest;
        }
      }
      d--;
      posib = posibs[d]; // backtrack ...
    }

    // copy the results into global memory
    uint_fast32_t old_idx;
    // fetch_and_add in OpenMP
#pragma omp atomic capture
    { old_idx = start_level3_cnt; start_level3_cnt += level4_cnt; } // atomically update start_level3_cnt, but capture original value of start_level3_cnt in old_idx
    memcpy(&start_level3[old_idx][0], &start_level4[0][0], level4_cnt*3*sizeof(uint_fast32_t));
  }

#pragma omp parallel for reduction(+ : num) schedule(dynamic)
  for (uint_fast32_t cnt = 0; cnt < start_level3_cnt; cnt++) {
    uint_fast32_t cols[MAXN], posibs[MAXN]; // Our backtracking 'stack'
    uint_fast32_t diagl[MAXN], diagr[MAXN];
    int8_t rest[MAXN]; // number of rows left
    int_fast16_t d = 1; // d is our depth in the backtrack stack
    // The UINT_FAST32_MAX here is used to fill all 'coloumn' bits after n ...
    cols[d] = start_level3[cnt][0];
    // This places the first two queens
    diagl[d] = start_level3[cnt][1];
    diagr[d] = start_level3[cnt][2];
#define LOOKAHEAD 3
    // we're allready two rows into the field here
    rest[d] = STORE_LEVEL;

    //  The variable posib contains the bitmask of possibilities we still have
    //  to try in a given row ...
    uint_fast32_t posib = (cols[d] | diagl[d] | diagr[d]);

    diagl[d] <<= 1;
    diagr[d] >>= 1;

    while (d > 0) {
      int8_t l_rest = rest[d];

      while (posib != UINT_FAST32_MAX) {
        // The standard trick for getting the rightmost bit in the mask
        uint_fast32_t bit = ~posib & (posib + 1);
        posib ^= bit; // Eliminate the tried possibility.
        uint_fast32_t new_diagl = (bit << 1) | diagl[d];
        uint_fast32_t new_diagr = (bit >> 1) | diagr[d];
        bit |= cols[d];
        uint_fast32_t new_posib = (bit | new_diagl | new_diagr);

        if (new_posib != UINT_FAST32_MAX) {
            uint_fast32_t lookahead1 = (bit | (new_diagl << (LOOKAHEAD - 2)) | (new_diagr >> (LOOKAHEAD - 2)));
            uint_fast32_t lookahead2 = (bit | (new_diagl << (LOOKAHEAD - 1)) | (new_diagr >> (LOOKAHEAD - 1)));
            uint_fast32_t allowed2 = l_rest > (int8_t)0;

            if(allowed2 && ((lookahead2 == UINT_FAST32_MAX) || (lookahead1 == UINT_FAST32_MAX))) {
                continue;
            }

          // The next two lines save stack depth + backtrack operations
          // when we passed the last possibility in a row.
          // Go lower in the stack, avoid branching by writing above the current
          // position
          posibs[d] = posib;
          d += posib != UINT_FAST32_MAX; // avoid branching with this trick
          posib = new_posib;

          l_rest--;

          // make values current
          cols[d] = bit;
          diagl[d] = new_diagl << 1;
          diagr[d] = new_diagr >> 1;
          rest[d] = l_rest;
        } else {
          // when all columns are used, we found a solution
          num += bit == UINT_FAST32_MAX;
        }
      }
      d--;
      posib = posibs[d]; // backtrack ...
    }
  }
  return num * 2;
}

// expected results from https://oeis.org/A000170
const uint64_t results[MAXN] = {
    1ULL,
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
    234907967154122528ULL,
    0,  // Not yet calculated
    0}; // Not yet calculated

void print_usage() {
  printf("usage: m-queens BOARDSIZE\n");
  printf("   or: m-queens START END\n");
  printf("   or: m-queens -h\n");
  printf("   or: m-queens --help\n");
  printf("\n");
  printf("\
This program computes the number of solutions for the n queens problem\n\
(see https://en.wikipedia.org/wiki/Eight_queens_puzzle) for the board size\n\
BOARDSIZE or for a range of board sizes beginning with START and ending\n\
with END.\n");
  printf("\n");
  printf("options:\n");
  printf("  -h, --help      print this usage information\n");
  printf("\n");
  printf("AUTHOR : sudden6 <sudden6@gmx.at>\n");
  printf("SOURCE : https://github.com/sudden6/m-queens\n");
  printf("LICENSE: GNU General Public License v3.0\n");
}

int main(int argc, char **argv) {

  int start = 2;
  int end = 16; // This should take a few minutes on a normal PC

  if (argc == 2) {
    if((strcmp("-h", argv[1]) == 0) || (strcmp("--help", argv[1]) == 0)) {
      print_usage();
      return  EXIT_SUCCESS;
    }

    start = atoi(argv[1]);
    if (start < 2 || start > MAXN) {
      printf("BOARDSIZE must be between 2 and %d!\n", MAXN);
      return EXIT_FAILURE;
    }
    end = start;
  }

  if (argc == 3) {
    start = atoi(argv[1]);
    if (start < 2 || start > MAXN) {
      printf("START must be between 2 and %d\n", MAXN);
      return EXIT_FAILURE;
    }
    end = atoi(argv[2]);
    if (end < start || end > MAXN) {
      printf("END must be between %d and %d\n", start, MAXN);
      return EXIT_FAILURE;
    }
  }

  // properly cast
  uint8_t start8 = (uint8_t) start;
  uint8_t end8 = (uint8_t) end;

  // track if all results passed
  int all_pass = 1;

  for (uint8_t n = start8; n <= end8; n++) {
    double time_diff, time_start; // for measuring calculation time
    time_start = get_time();
    uint64_t result = nqueens(n);
    time_diff = (get_time() - time_start); // calculating time difference
    // check if result is correct
    int pass = result == results[n - 1];
    all_pass &= pass;
    pass ? printf("PASS ") : printf("FAIL ");
    printf("N %2d, Solutions %18" PRIu64 ", Expected %18" PRIu64
           ", Time %f sec., Solutions/s %f\n",
           n, result, results[n - 1], time_diff, result / time_diff);
  }

  return all_pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
