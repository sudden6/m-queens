#include <climits>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iterator>
#include <thread>
#include "clsolver.h"
#include "cpusolver.h"
#include "solverstructs.h"
#include "presolver.h"
#include "cxxopts.hpp"
#include "result_file.h"
#include "start_file.h"

#include <vector>
#include <chrono>
#include <regex>

constexpr uint8_t MAXN = 29;
constexpr uint8_t MINN = 4;
constexpr uint8_t MIN_PRE_PLACED = 3;

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

static void solve_from_range(ISolver& solver, uint8_t start, uint8_t end) {
    for (uint8_t i = start; i <= end; i++) {
        if(!solver.init(i, 3)) {
            std::cout << "Solver init failed" << std::endl;
            return;
        }

        uint64_t result = 0;
        auto time_start = std::chrono::high_resolution_clock::now();
        std::vector<start_condition> st = PreSolver::create_preplacement(i);

        result = solver.solve_subboard(st);
        auto time_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = time_end - time_start;

        std::cout.precision(3);

        result == results[i - 1] ? printf("PASS ") : printf("FAIL ");
        std::cout << "N " << std::to_string(i)
                << ", Solutions " << std::to_string(result)
                << ", Expected " << std::to_string(results[i - 1])
                << ", Time " << std::to_string(elapsed.count())
                << ", Solutions/s " << result/elapsed.count()
                << std::endl;
    }

    std::cout << "DONE" << std::endl;
}

static void solve_from_file(ISolver& solver, const std::string& filename) {
    start_file::file_info fi = start_file::parse_filename(filename);

    if (fi.boardsize == 0) {
        exit(EXIT_FAILURE);
    }

    if (fi.boardsize < MINN || fi.boardsize > MAXN) {
        std::cout << "Boardsize out of range" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (fi.placed < MIN_PRE_PLACED || fi.placed >= fi.boardsize) {
        std::cout << "Invalid number of placed queens" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (fi.start_idx > fi.end_idx) {
        std::cout << "Invalid range" << std::endl;
        exit(EXIT_FAILURE);
    }

    uint64_t start_count = fi.end_idx - fi.start_idx + 1;

    std::vector<start_condition_t> start = start_file::load_all(filename);

    if(start_count != start.size()) {
        std::cout << "File content not matching description" << std::endl;
        exit(EXIT_FAILURE);
    }

    if(!solver.init(fi.boardsize, fi.placed)) {
        std::cout << "Failed to initialize solver" << std::endl;
        exit(EXIT_FAILURE);
    }

    uint64_t result = 0;
    auto time_start = std::chrono::high_resolution_clock::now();
    result = solver.solve_subboard(start);
    auto time_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = time_end - time_start;

    std::cout << "N " << std::to_string(fi.boardsize)
              << ", Placed " << std::to_string(fi.placed)
              << ", Range [" << std::to_string(fi.start_idx)
              << ":" << std::to_string(fi.end_idx)
              << "], Solutions " << std::to_string(result)
              << ", Time " << std::to_string(elapsed.count())
              << ", Solutions/s " << std::to_string(result/elapsed.count())
              << std::endl;

    std::string res_filname = filename.substr(0, filename.size() - 4) + ".res";

    std::vector<uint64_t> results = {result};

    if(!result_file::save(results, res_filname)) {
        std::cout << "Failed to save result" << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char **argv) {
    uint8_t start = 0;
    uint8_t end = 0;
    bool solve_range = false;
    bool list_opencl = false;
    bool presolve_file = false;
    bool help = false;
    bool debug = false;
    unsigned int ocl_platform = 0;
    unsigned int ocl_device = 0;
    std::string solver_string = "";
    std::string presolve_file_name = "";
    try
    {
      cxxopts::Options options("m-queens2", " - a CPU and GPU solver for the N queens problem");
      options.add_options()
        ("s,start", "[4..29] boardsize to solve", cxxopts::value(start))
        ("e,end", "[OPTIONAL] end value to solve a range of board sizes", cxxopts::value(end))
        ("l,list", "list and enumerate available OpenCL devices, must be the only option passed", cxxopts::value(list_opencl))
        ("m,mode", "solve on [cpu] or OpenCL [ocl] mode", cxxopts::value(solver_string)->default_value("ocl"))
        ("p,platform", "OpenCL platform to use", cxxopts::value(ocl_platform)->default_value("0"))
        ("d,device", "OpenCL device to use", cxxopts::value(ocl_device)->default_value("0"))
        ("f,file", "Use presolve file generated by 'presolver', '-s' and '-e' are invalid in this mode", cxxopts::value(presolve_file_name)->default_value(""))
        ("ocl_debug", "Print detailed debug messages from OpenCL kernel, might produce a wall of text", cxxopts::value(debug))
        ("h,help", "Print this information", cxxopts::value(help))
        ;

      auto result = options.parse(argc, argv);
      if (result.arguments().size() == 0) {
          help = true;
      }

      if (help)
      {
        std::cout << options.help();
        exit(EXIT_SUCCESS);
      }

      if (list_opencl)
      {
        std::cout << "The following OpenCL devices are available:" << std::endl;
        ClSolver::enumerate_devices();
        exit(EXIT_SUCCESS);
      }

      if (result.count("file") > 0) {
          if (result.count("file") != 1) {
              std::cout << "Only one presolve file supported" << std::endl;
              exit(EXIT_FAILURE);
          }

          if(result.count("start") || result.count("end")) {
              std::cout << "[start] and [end] are not supported with presolve files" << std::endl;
              exit(EXIT_FAILURE);
          }
          presolve_file = true;
      } else {
          if (!result.count("start"))
          {
            std::cout << options.help() << std::endl;
            exit(EXIT_FAILURE);
          }

          if (result.count("end"))
          {
            solve_range = true;
          }
      }

    } catch (const cxxopts::OptionException& e)
    {
      std::cout << "error parsing options: " << e.what() << std::endl;
      exit(EXIT_FAILURE);
    }

    ISolver* solver = nullptr;

    if(solver_string == "CPU" || solver_string == "cpu") {
        solver = new cpuSolver();
    } else if(solver_string == "OCL" || solver_string == "ocl") {
        solver = ClSolver::makeClSolver(ocl_platform, ocl_device, debug);
        if(!solver) {
            exit(EXIT_FAILURE);
        }
    } else {
        std::cout << "[type] must be either CPU or OCL" << std::endl;
        exit(EXIT_FAILURE);
    }

    if(solver == nullptr) {
        std::cout << "Couldn't start solver" << std::endl;
        exit(EXIT_FAILURE);
    }

    if(presolve_file) {
        solve_from_file(*solver, presolve_file_name);
    } else {
        if(start <= MINN || start > MAXN) {
          std::cout << "[start] must be greater 4 and smaller than " << std::to_string(MAXN) << std::endl;
          exit(EXIT_FAILURE);
        }

        if(solve_range) {
          if(end < start || start > MAXN) {
            std::cout << "[end] must be equal or greater [start] and smaller than " << std::to_string(MAXN) << std::endl;
            exit(EXIT_FAILURE);
          }
        } else {
            end = start;
        }

        solve_from_range(*solver, start, end);
    }

  return 0;
}

