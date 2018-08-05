#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <iostream>
#include <iterator>
#include <thread>
#include <time.h>
#include "clsolver.h"
#include "cpusolver.h"
#include "solverstructs.h"
#include "presolver.h"

#include <vector>

// uncomment to start with n=2 and compare to known results
//#define TESTSUITE

#ifndef N
#define N 18
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

std::vector<start_condition> create_preplacement(uint_fast8_t n) {
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
        //result[start_cnt].placed = 2;
        start_cnt++;
      }
    }
    result.resize(start_cnt); // shrink
    return result;
}

std::vector<start_condition> create_subboards(uint_fast8_t n, uint_fast8_t placed, uint_fast8_t depth, start_condition& start) {
    std::vector<start_condition> result;

    if(n < 2) {
        return  result;
    }

    if(depth == 0) {
        result.push_back(start);
        return  result;
    }

    // compute maximum size needed for storing all results
    uint_fast32_t num_starts = 1;
    //uint_fast8_t new_depth = start.placed + depth;
    int_fast8_t start_placed = n - placed;
    // ensure we don't preplace all rows
    if(start_placed - depth < 1) {
        result.push_back(start);
        return result;
    }

    for(int i = 0; i < depth; i++) {
        num_starts *= start_placed - i;
    }


    uint_fast32_t res_cnt = 0;
    result.resize(num_starts);         // preallocate memory

    uint_fast32_t cols[MAXN], posibs[MAXN]; // Our backtracking 'stack'
    uint_fast32_t diagl[MAXN], diagr[MAXN];
    int_fast8_t rest[MAXN]; // number of rows left
    int_fast16_t d = 0; // d is our depth in the backtrack stack
    // The UINT_FAST32_MAX here is used to fill all 'coloumn' bits after n ...
    cols[d] = start.cols | (UINT_FAST32_MAX << n);
    // This places the first two queens
    diagl[d] = start.diagl;
    diagr[d] = start.diagr;
#define LOOKAHEAD 3
    // we're allready two rows into the field here
    rest[d] = n - LOOKAHEAD - placed;
    const int_fast8_t max_depth = rest[d] - depth + 1;  // save result at this depth

    //  The variable posib contains the bitmask of possibilities we still have
    //  to try in a given row ...
    uint_fast32_t posib = (cols[d] | diagl[d] | diagr[d]);

    while (d >= 0) {
      // moving the two shifts out of the inner loop slightly improves
      // performance
      uint_fast32_t diagl_shifted = diagl[d] << 1;
      uint_fast32_t diagr_shifted = diagr[d] >> 1;
      int_fast8_t l_rest = rest[d];

      while (posib != UINT_FAST32_MAX) {
        // The standard trick for getting the rightmost bit in the mask
        uint_fast32_t bit = ~posib & (posib + 1);
        uint_fast32_t new_diagl = (bit << 1) | diagl_shifted;
        uint_fast32_t new_diagr = (bit >> 1) | diagr_shifted;
        uint_fast32_t new_posib = (cols[d] | bit | new_diagl | new_diagr);
        posib ^= bit; // Eliminate the tried possibility.
        bit |= cols[d];

        if (new_posib != UINT_FAST32_MAX) {
            uint_fast32_t lookahead1 = (bit | (new_diagl << (LOOKAHEAD - 2)) | (new_diagr >> (LOOKAHEAD - 2)));
            uint_fast32_t lookahead2 = (bit | (new_diagl << (LOOKAHEAD - 1)) | (new_diagr >> (LOOKAHEAD - 1)));
            uint_fast8_t allowed1 = l_rest >= 0;
            uint_fast8_t allowed2 = l_rest > 0;

            if(allowed1 && (lookahead1 == UINT_FAST32_MAX)) {
                continue;
            }

            if(allowed2 && (lookahead2 == UINT_FAST32_MAX)) {
                continue;
            }

            if(l_rest == max_depth) {
                result[res_cnt].cols = static_cast<uint32_t> (bit);
                result[res_cnt].diagl = static_cast<uint32_t> (new_diagl);
                result[res_cnt].diagr = static_cast<uint32_t>(new_diagr);
                res_cnt++;
                continue;
            }

            l_rest--;

            // The next two lines save stack depth + backtrack operations
            // when we passed the last possibility in a row.
            // Go lower in the stack, avoid branching by writing above the current
            // position
            posibs[d + 1] = posib;
            d += posib != UINT_FAST32_MAX; // avoid branching with this trick
            posib = new_posib;

            // make values current
            cols[d] = bit;
            diagl[d] = new_diagl;
            diagr[d] = new_diagr;
            rest[d] = l_rest;
            diagl_shifted = new_diagl << 1;
            diagr_shifted = new_diagr >> 1;
        }
      }
      posib = posibs[d]; // backtrack ...
      d--;
    }

    result.resize(res_cnt); // shrink
    return result;
}



// expected results from https://oeis.org/A000170
static const uint64_t results[27] = {
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
    234907967154122528ULL};

constexpr uint32_t THREADS = 1;
static uint64_t thread_results[THREADS] = {0};

void thread_worker(ClSolver solver, uint32_t id,
                   std::vector<start_condition>::iterator begin,
                   std::vector<start_condition>::iterator end) {
    std::cout << "Starting thread: " << std::to_string(id) << std::endl;

    std::vector<start_condition> thread_batch(begin, end);

    thread_results[id] = solver.solve_subboard(thread_batch);
}

int main(int argc, char **argv) {

#ifdef TESTSUITE
  int i = 8;
#else
  int i = N;
#endif
  if (argc == 2) {
    i = atoi(argv[1]);
    if (i < 1 || i > MAXN) {
      printf("n must be between 2 and %d!\n", MAXN);
    }
  }

  cpuSolver cpu;
  ClSolver ocl;


  for (; i <= N; i++) {
    double time_diff, time_start; // for measuring calculation time
    cpu.init(i, 2);
    ClSolver ocl[THREADS] = {};
    std::thread* threads[THREADS] = {nullptr};

    cpu.init(i, 2);
    for(uint32_t t = 0; t < THREADS; t++) {
        ocl[t].init(i, 2);
    }
    uint64_t result = 0;
    time_start = get_time();
    std::vector<start_condition> st = create_preplacement(i);

    // round up
    const size_t block_size = (st.size() + THREADS - 1)/THREADS;

    auto block_start = st.begin();

    for(uint32_t t = 0; t < THREADS; t++) {
        auto end = block_start;
        size_t remaining(std::distance(end, st.end()));
        size_t step = block_size;
        if (remaining < step)
        {
          step = remaining;
        }
        std::advance(end, step);
        threads[t] = new std::thread(thread_worker, std::move(ocl[t]), t, block_start, end);
        block_start = end;
    }

    //result = ocl.solve_subboard(st);

    result = 0;
    for(uint32_t t = 0; t < THREADS; t++) {
        threads[t]->join();
        result += thread_results[t];
    }

    for(uint32_t t = 0; t < THREADS; t++) {
        delete threads[t];
    }
    /*
    size_t st_size = st.size();
    for(size_t j = 0; j < st_size; j++) {
        //std::cout << j << " of " << st_size;
        //std::vector<start_condition> second = create_subboards(i, 2, depth, st[j]);

        uint64_t ocl_res;
        std::vector<start_condition> second = {st[j]};
        //uint64_t cpu_res = cpu.solve_subboard(second);
        ocl_res = ocl.solve_subboard(second);
        //std::cout << " subboards: " << second.size() << " DONE" << std::endl;
        /*
        if(cpu_res != ocl_res) {
            std::cout << "Result mismatch" << std::endl;
        }//
        result += ocl_res;
    }*/

    time_diff = (get_time() - time_start); // calculating time difference
    result == results[i - 1] ? printf("PASS ") : printf("FAIL ");
    std::cout << "N " << i << ", Solutions " << result << ", Expected " << results[i - 1] <<
           ", Time " << time_diff << " , Solutions/s " << result/time_diff << std::endl;
  }

  std::cout << "DONE" << std::endl;
  return 0;
}
