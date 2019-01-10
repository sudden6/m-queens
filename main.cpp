#include <climits>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iterator>
#include <thread>
#include <time.h>
#include "clsolver.h"
#include "cpusolver.h"
#include "solverstructs.h"
#include "presolver.h"
#include "cxxopts.hpp"

#include <vector>

#define MAXN 29


#ifdef _MSC_VER
double get_time() {
    return GetTickCount64() / 1000.0;
}

#else
#include <sys/time.h>

// get the current wall clock time in seconds
double get_time() {
  struct timeval tp;
  gettimeofday(&tp, nullptr);
  return tp.tv_sec + tp.tv_usec / 1000000.0;
}

#endif

/**
 * @brief compute the maximum number of possibilities at a specified depth
 * @param current number of possibilities at the current depth
 * @param placed already placed queens
 * @param boardsize size of the board
 * @param depth number of steps to calculate
 * @return number of possibilites at this depth
 */
size_t possibs_at_depth(size_t current, uint8_t placed, uint8_t boardsize, uint8_t depth) {
    size_t result = current;  // depends on preplacement algorithm
    for(uint8_t i = 0; i < depth; i++) {
        result *= boardsize - 1 - placed - i;
    }
    return result;
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

    // maximum number of start possibilities at row 3
    result.resize(possibs_at_depth(start_cnt, 2, n, 1));

    uint_fast32_t start_level3_cnt = 0;
  #define START_LEVEL4 (MAXN - 4)

  #pragma omp parallel for schedule(dynamic)
    for (uint_fast16_t cnt = 0; cnt < start_cnt; cnt++) {
      start_condition_t start_level4[START_LEVEL4];
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
              start_level4[level4_cnt].cols = bit;
              start_level4[level4_cnt].diagl = new_diagl;
              start_level4[level4_cnt].diagr = new_diagr;
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
      memcpy(&result[old_idx], &start_level4[0], level4_cnt*sizeof(start_condition_t));
    }

    result.resize(start_level3_cnt); // shrink
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

int main(int argc, char **argv) {
    long start = 0;
    long end = 0;
    bool solve_range = false;
    bool list_opencl = false;
    bool help = false;
    unsigned int ocl_platform = 0;
    unsigned int ocl_device = 0;
    std::string solver_string = "";
    try
    {
      cxxopts::Options options("m-queens2", " - a CPU and GPU solver for the N queens problem");
      options.add_options()
        ("s,start", "[4..29] start value for N", cxxopts::value(start))
        ("e,end", "[OPTIONAL] end value to solve a range of N values", cxxopts::value(end))
        ("l,list", "list and enumerate available OpenCL devices, must be the only option passed", cxxopts::value(list_opencl))
        ("m,mode", "solve on [cpu] or OpenCL [ocl] mode", cxxopts::value(solver_string)->default_value("ocl"))
        ("p,platform", "OpenCL platform to use", cxxopts::value(ocl_platform)->default_value("0"))
        ("d,device", "OpenCL device to use", cxxopts::value(ocl_device)->default_value("0"))
        ("h,help", "Print this information")
        ;

      auto result = options.parse(argc, argv);

      if (help)
      {
        options.help();
        exit(0);
      }

      if (list_opencl)
      {
        std::cout << "The following OpenCL devices are available:" << std::endl;
        ClSolver::enumerate_devices();
        exit(0);
      }

      if (!result.count("start"))
      {
        std::cout << options.help() << std::endl;
        exit(1);
      }

      if (result.count("end"))
      {
        solve_range = true;
      }

    } catch (const cxxopts::OptionException& e)
    {
      std::cout << "error parsing options: " << e.what() << std::endl;
      exit(1);
    }

    if(start <= 4 || start > MAXN) {
      std::cout << "[start] must be greater 4 and smaller than " << std::to_string(MAXN) << std::endl;
      exit(1);
    }

    if(solve_range) {
      if(end < start || start > MAXN) {
        std::cout << "[end] must be equal or greater [start] and smaller than " << std::to_string(MAXN) << std::endl;
        exit(1);
      }
    } else {
        end = start;
    }

    ISolver* solver = nullptr;

    if(solver_string == "CPU" || solver_string == "cpu") {
        solver = new cpuSolver();
    } else if(solver_string == "OCL" || solver_string == "ocl") {
        solver = ClSolver::makeClSolver(ocl_platform, ocl_device);
        if(!solver) {
            exit(1);
        }
    } else {
        std::cout << "[type] must be either CPU or OCL" << std::endl;
        exit(1);
    }

  uint8_t u8_start = static_cast<uint8_t>(start);
  uint8_t u8_end = static_cast<uint8_t>(end);

  for (uint8_t i = u8_start; i <= u8_end; i++) {
    double time_diff, time_start; // for measuring calculation time

    solver->init(i, 3);

    uint64_t result = 0;
    time_start = get_time();
    std::vector<start_condition> st = create_preplacement(i);

    result = solver->solve_subboard(st);

    time_diff = (get_time() - time_start); // calculating time difference
    result == results[i - 1] ? printf("PASS ") : printf("FAIL ");
    std::cout << "N " << std::to_string(i)
              << ", Solutions " << std::to_string(result)
              << ", Expected " << std::to_string(results[i - 1])
              << ", Time " << std::to_string(time_diff)
              << ", Solutions/s " << std::to_string(result/time_diff)
              << std::endl;
  }

  std::cout << "DONE" << std::endl;
  return 0;
}
