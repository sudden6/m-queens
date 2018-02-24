#include "presolver.h"

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

    d = 0;
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

    // detect if no possibilities left
    if (posib == UINT_FAST32_MAX) {
        valid = false;
    }

}

std::vector<start_condition> PreSolver::getNext(size_t count)
{
    std::vector<start_condition> result;

    if(!valid) {
        return result;
    }

    if(depth == 0) {
        valid = false;
        result.push_back(start);
        return result;
    }

    uint_fast32_t res_cnt = 0;
    result.resize(count);         // preallocate memory

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
                if(res_cnt < count) {
                    continue;
                } else {
                    return result;
                }
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

    valid = false;  // all subboards created
    result.resize(res_cnt); // shrink
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

    valid = false;  // all subboards created
    return it;
}

bool PreSolver::empty() const
{
    return !valid;
}
