#ifndef PRESOLVER_H
#define PRESOLVER_H

#include "solverstructs.h"

#include <vector>

class PreSolver
{
public:
    PreSolver();
    PreSolver(uint_fast8_t n, uint_fast8_t placed, uint_fast8_t depth, start_condition start);
    std::vector<start_condition> getNext(size_t count);
    std::vector<start_condition>::iterator getNext(std::vector<start_condition>::iterator it,
                 const std::vector<start_condition>::const_iterator end);
    bool empty() const;
    std::vector<uint8_t> save() const;
    bool load(std::vector<uint8_t> data);

    static size_t possibs_at_depth(size_t current, uint8_t placed, uint8_t boardsize, uint8_t depth);
    static std::vector<start_condition> create_preplacement(uint_fast8_t n);
private:
    uint_fast8_t n = 0;
    uint_fast8_t placed = 0;
    uint_fast8_t depth = 0;
    bool valid = false;
    start_condition start;

    static constexpr unsigned MAXD = 20;

    // maximum board size for which this solver is designed
    static constexpr unsigned MAXN = 29;

    // Our backtracking 'stack'
    uint_fast32_t cols[MAXD], posibs[MAXD];
    uint_fast32_t diagl[MAXD], diagr[MAXD];
    int_fast8_t rest[MAXD]; // number of rows left
    int_fast16_t d = 0; // d is our depth in the backtrack stack
    //  The variable posib contains the bitmask of possibilities we still have
    //  to try in a given row ...
    uint_fast32_t posib = 0;

    int_fast8_t max_depth = 0;

    // struct definition for class serialization
    struct bin_save {
        uint_fast8_t n = 0;
        uint_fast8_t placed = 0;
        uint_fast8_t depth = 0;
        bool valid = false;
        start_condition start;
        uint_fast32_t cols[PreSolver::MAXD], posibs[PreSolver::MAXD];
        uint_fast32_t diagl[PreSolver::MAXD], diagr[PreSolver::MAXD];
        int_fast8_t rest[MAXD];
        uint_fast32_t posib = 0;
        int_fast8_t max_depth = 0;
    };
};

#endif // PRESOLVER_H
