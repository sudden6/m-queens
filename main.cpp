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
#include "filereader.h"
#include "solverstructs.h"
#include "presolver.h"
#include "cxxopts.hpp"

#include <vector>

// uncomment to start with n=2 and compare to known results
#define TESTSUITE

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
        uint_fast32_t bit0 = UINT32_C(1) << q0;
        uint_fast32_t bit1 = UINT32_C(1) << q1;
        result[start_cnt].cols = static_cast<uint32_t>(bit0 | bit1);
        result[start_cnt].diagl = static_cast<uint32_t>((bit0 << 2) | (bit1 << 1));
        result[start_cnt].diagr = static_cast<uint32_t>((bit0 >> 2) | (bit1 >> 1));
        start_cnt++;
      }
    }
    result.resize(start_cnt); // shrink
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
    std::string filename_none{"./N_5_NONE.dat2"};
    std::string filename_rot{"./N_5_ROTATE.dat2"};
    FileReader file_none(filename_none);
    FileReader file_rot(filename_none);

    cpuSolver testSolver;
    testSolver.init(5, 0);

    uint64_t res_cnt = testSolver.solve_subboard(file_none.getNext(2)) * 8;
    res_cnt += testSolver.solve_subboard(file_rot.getNext(1)) * 2;

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

    solver->init(i, 2);

    uint64_t result = 0;
    time_start = get_time();
    std::vector<start_condition> st = create_preplacement(i);

    result = solver->solve_subboard(st);

    time_diff = (get_time() - time_start); // calculating time difference
    result == results[i - 1] ? printf("PASS ") : printf("FAIL ");
    std::cout << "N " << i << ", Solutions " << result << ", Expected " << results[i - 1] <<
           ", Time " << time_diff << " , Solutions/s " << result/time_diff << std::endl;
  }

  std::cout << "DONE" << std::endl;
  return 0;
}
