#ifndef CLSOLVER_H
#define CLSOLVER_H

#include <cstdint>
#include <list>
#include <memory>
#include <vector>
#include "solverstructs.h"

#include <CL/cl.hpp>

class ClSolver
{
public:
    ClSolver();
    bool init(uint8_t boardsize, uint8_t placed);
    uint64_t solve_subboard(const std::vector<start_condition>::const_iterator &begin,
                            const std::vector<start_condition>::const_iterator &end);
private:

    enum class STAGE_TYPE {FIRST, MID, LAST};

    struct sieve_stage {
        cl::Program program;
        cl::Kernel clKernel;
        cl::Buffer clStageBuf;  // buffer for the data alive after this stage
        cl::Buffer clFillCount; // buffer that holds the fill status
        cl::Buffer clSum;       // buffer for the sum at the last stage
        cl::Event clStageDone;  // event when this stage is complete
        //std::unique_ptr<cl_int> hostFillCount = nullptr;
        STAGE_TYPE type;        // type of the current stage
        uint8_t index;          // index of the current stage, mainly for debugging
        uint8_t placed;         // number of queens at the beginning of this stage
        uint32_t expansion;     // maximum number of solutions generated from one input solution
        uint8_t depth;          // number of queens placed in this stage
        cl_int buf_threshold;   // limit after which the buffer is emptied
        uint32_t max_fill = 0;  // maximum fill level of the stage/sum buffer
    };

    struct stage_work_item {
        cl_uint bufferIdx;      // must be 32 Bit, to match OpenCL kernel
        uint32_t stageIdx;
        cl_int taken;           // amount of items marked as taken, but not yet removed from buffer
        stage_work_item(cl_uint bufferIdx, uint32_t stageIdx, cl_int taken)
            : bufferIdx{bufferIdx}
            , stageIdx{stageIdx}
            , taken{taken}
        {}
    };


    uint8_t presolve_depth = 0;
    uint8_t placed = 0;
    uint8_t boardsize = 0;
    cl::Context context;
    cl::Device device;
    cl::CommandQueue queue;
    std::vector<sieve_stage> stages;
    std::string sourceStr;
    bool first_stage();
    bool build_program(sieve_stage &stage);
    void compute_expansion(sieve_stage &stage);
    uint8_t queens_left();
    bool append_mid_stage();
    bool append_last_stage();
    void compute_buf_threshold(sieve_stage &stage);
    void fill_work_queue(cl::CommandQueue &queue, std::list<stage_work_item> &work_queue, sieve_stage &stage, cl_int threshold);
};

#endif // CLSOLVER_H
