#include "presolver.h"

#include <cstring>
#include <iterator>

PreSolver::PreSolver()
    : valid{false}
{
}

/**
 * @brief PreSolver::PreSolver
 * @param n board size
 * @param placed number of already placed queens
 * @param depth queens to place until done
 * @param start initial condition
 */
PreSolver::PreSolver(uint_fast8_t n, uint_fast8_t placed, uint_fast8_t depth, start_condition start)
{
    if(n < 2) {
        return;
    }

    if(placed < 1) {
        return;
    }

    if(depth > MAXD) {
        return;
    }

    this->n = n;
    this->placed = placed;
    this->depth = depth;
    this->start = start;
    this->valid = true;

    d = 1;
    // The UINT_FAST32_MAX here is used to fill all 'coloumn' bits after n ...
    cols[d] = start.cols | (UINT_FAST32_MAX << n);
    // This places the first two queens
    diagl[d] = start.diagl;
    diagr[d] = start.diagr;

#define LOOKAHEAD 3
    // we're allready two rows into the field here
    rest[d] = n - LOOKAHEAD - placed;
    max_depth = rest[d] - depth + 1;  // save result at this depth

    //  The variable posib contains the bitmask of possibilities we still have
    //  to try in a given row ...
    posib = (cols[d] | diagl[d] | diagr[d]);

    diagl[d] <<= 1;
    diagr[d] >>= 1;

    // detect if no possibilities left
    if (posib == UINT_FAST32_MAX) {
        valid = false;
    }

}

std::vector<start_condition> PreSolver::getNext(size_t count)
{
    std::vector<start_condition> result;
    result.resize(count);
    std::vector<start_condition>::iterator end = getNext(result.begin(), result.cend());
    result.resize(std::distance(result.begin(), end));
    return result;
}

std::vector<start_condition>::iterator PreSolver::getNext(std::vector<start_condition>::iterator it,
                        const std::vector<start_condition>::const_iterator end)
{
    if(it == end) {
        return it;
    }

    if(!valid) {
        return it;
    }

    if(depth == 0) {
        valid = false;
        *it = start;
        it++;
        return it;
    }

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


                if(l_rest == max_depth) {
                    it->cols = static_cast<uint32_t> (bit);
                    it->diagl = static_cast<uint32_t> (new_diagl);
                    it->diagr = static_cast<uint32_t>(new_diagr);
                    it++;
                    if(it != end) {
                        continue;
                    } else {
                        return it;
                    }
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

    valid = false;  // all subboards created
    return it;
}

bool PreSolver::empty() const
{
    return !valid;
}

std::vector<uint8_t> PreSolver::save() const
{
    struct bin_save save_data;
    save_data.n = this->n;
    save_data.placed = this->placed;
    save_data.depth = this->depth;
    save_data.valid = this->valid;
    save_data.start = this->start;
    memcpy(save_data.cols, this->cols, sizeof(uint_fast32_t) * MAXD);
    memcpy(save_data.posibs, this->posibs, sizeof(uint_fast32_t) * MAXD);
    memcpy(save_data.diagl, this->diagl, sizeof(uint_fast32_t) * MAXD);
    memcpy(save_data.diagr, this->diagr, sizeof(uint_fast32_t) * MAXD);
    memcpy(save_data.rest, this->rest, sizeof(int_fast8_t) * MAXD);
    save_data.posib = this->posib;
    save_data.max_depth = this->max_depth;

    std::vector<uint8_t> result;
    result.resize(sizeof(save_data));
    memcpy(result.data(), &save_data, result.size());

    return result;
}

bool PreSolver::load(std::vector<uint8_t> data)
{
    struct bin_save load_data;
    if(data.size() != sizeof(load_data)) {
        return false;
    }

    memcpy(&load_data, data.data(), data.size());

    this->n = load_data.n;
    this->placed = load_data.placed;
    this->depth = load_data.depth;
    this->valid = load_data.valid;
    this->start = load_data.start;
    memcpy(this->cols, load_data.cols, sizeof(uint_fast32_t) * MAXD);
    memcpy(this->posibs, load_data.posibs, sizeof(uint_fast32_t) * MAXD);
    memcpy(this->diagl, load_data.diagl, sizeof(uint_fast32_t) * MAXD);
    memcpy(this->diagr, load_data.diagr, sizeof(uint_fast32_t) * MAXD);
    memcpy(this->rest, load_data.rest, sizeof(int_fast8_t) * MAXD);
    this->posib = load_data.posib;
    this->max_depth = load_data.max_depth;

    return true;
}
