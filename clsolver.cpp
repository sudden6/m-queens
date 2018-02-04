#include "clsolver.h"
#include <ios>
#include <iostream>
#include <iterator>
#include <string>
#include <fstream>
#include <streambuf>
#include <sstream>
#include <CL/cl2.hpp>

ClSolver::ClSolver()
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if(platforms.size() == 0) {
        std::cout << "No OpenCL platforms found" << std::endl;
        return;
    }

    cl::Platform plat = platforms[0];

    std::cout << "Platform name: " << plat.getInfo<CL_PLATFORM_NAME>() << std::endl;
    std::cout << "Platform version: " << plat.getInfo<CL_PLATFORM_VERSION>() << std::endl;

    if (plat() == nullptr)  {
        std::cout << "No OpenCL 2. platform found.";
        return;
    }

    // load source code
    std::ifstream sourcefile("clqueens.cl");
    std::string sourceStr((std::istreambuf_iterator<char>(sourcefile)),
                     std::istreambuf_iterator<char>());

    // create OpenCL program
    cl::Program clQueensProg(sourceStr);

    program = cl::Program(sourceStr);
}

uint64_t ClSolver::solve_subboard(uint_fast8_t n, std::vector<start_condition>& start)
{
    std::ostringstream optionsStream;
    optionsStream << "-DN=" << n;
    std::string options = optionsStream.str();
    program.build("-DN=5");
    std::cout << "OpenCL build log:" << std::endl;
    auto buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>();
    for (auto &pair : buildlog) {
        std::cout << pair.second << std::endl;
    }

    // Create command queue.
    cl::CommandQueue queue{};
   cl::Kernel solve_subboard(program, "solve_subboard");
   std::vector<cl_ulong> results(start.size(), 0);

   // Allocate device buffers and transfer input data to device.
   cl::Buffer start_buf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
       start.size() * sizeof(start_condition), start.data());

   cl::Buffer results_buf(CL_MEM_READ_WRITE,
       results.size() * sizeof(cl_ulong), results.data());

   // Set kernel parameters.
   solve_subboard.setArg(0, start_buf);
   solve_subboard.setArg(1, results_buf);

   // Launch kernel on the compute device.
   queue.enqueueNDRangeKernel(solve_subboard, cl::NullRange, start.size(), cl::NullRange);
   // Get result back to host.
   queue.enqueueReadBuffer(results_buf, CL_TRUE, 0, results.size() * sizeof(cl_ulong), results.data());

   queue.finish();
   for(auto a : results) {
       std::cout << "test: " << a << std::endl;
   }
}



