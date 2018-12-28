#include "cpusolver.h"
#include "cpusolver.h"
#include <iostream>
#include <cstdint>

cpuSolver::cpuSolver()
{

}

constexpr uint_fast8_t MINN = 2;
constexpr uint_fast8_t MAXN = 29;

bool cpuSolver::init(uint8_t boardsize, uint8_t placed)
{
    if(boardsize > MAXN || boardsize < MINN) {
        std::cout << "Invalid boardsize for cpusolver" << std::endl;
        return false;
    }

    if(placed >= boardsize) {
        std::cout << "Invalid number of placed queens for cpusolver" << std::endl;
        return false;
    }
    this->boardsize = boardsize;
    this->placed = placed;

    return true;
}


uint64_t cpuSolver::solve_subboard(const std::vector<start_condition>& starts) {

  // counter for the number of solutions
  // sufficient until n=29
  uint_fast64_t num = 0;
  size_t start_cnt = starts.size();

#pragma omp parallel for reduction(+ : num) schedule(dynamic)
  for (size_t cnt = 0; cnt < start_cnt; cnt++) {
    uint_fast64_t l_num = 0;
    uint_fast32_t cols[MAXN], posibs[MAXN]; // Our backtracking 'stack'
    uint_fast32_t diagl[MAXN], diagr[MAXN];
    int_fast8_t rest[MAXN]; // number of rows left
    int_fast16_t d = 0; // d is our depth in the backtrack stack
    // The UINT_FAST32_MAX here is used to fill all 'coloumn' bits after n ...
    cols[d] = starts[cnt].cols | (UINT_FAST32_MAX << boardsize);
    // This places the first two queens
    diagl[d] = starts[cnt].diagl;
    diagr[d] = starts[cnt].diagr;
#define LOOKAHEAD 3
    // we're allready two rows into the field here
    rest[d] = boardsize - LOOKAHEAD - this->placed;

    //printf("cpuid: X, cols: %x, diagl: %x, diagr: %x, rest: %d\n", cols[d], diagl[d], diagr[d], rest[d]);

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

            // The next two lines save stack depth + backtrack operations
            // when we passed the last possibility in a row.
            // Go lower in the stack, avoid branching by writing above the current
            // position
            posibs[d + 1] = posib;
            d += posib != UINT_FAST32_MAX; // avoid branching with this trick
            posib = new_posib;

            l_rest--;

            // make values current
            cols[d] = bit;
            diagl[d] = new_diagl;
            diagr[d] = new_diagr;
            rest[d] = l_rest;
            diagl_shifted = new_diagl << 1;
            diagr_shifted = new_diagr >> 1;
        } else {
            // when all columns are used, we found a solution
            l_num += bit == UINT_FAST32_MAX;
        }
      }
      posib = posibs[d]; // backtrack ...
      d--;
    }
     num += l_num;
  }
  return num * 2;
}

void place_queen(uint_fast32_t& rows, uint_fast32_t& cols, uint_fast64_t& diags_u, uint_fast64_t& diags_d, uint8_t x, uint8_t y, uint8_t N) {
    if(x == 0xFF || y == 0xFF) {
        // no queen at this coordinates
        return;
    }

    rows |= UINT32_C(1) << y;
    cols |= UINT32_C(1) << x;
    diags_u |= UINT64_C(1) << (N - 1 + x - y);
    diags_d |= UINT64_C(1) << (x + y);
}

void load_preplacement(uint_fast32_t& rows, uint_fast32_t& cols, uint_fast64_t& diags_u, uint_fast64_t& diags_d, const Preplacement& pre, uint8_t N) {
    uint8_t col_i[4] = {0, 1, static_cast<uint8_t>(N - 2), static_cast<uint8_t>(N - 1)};
    uint8_t row_i[4] = {0, 1, static_cast<uint8_t>(N - 2), static_cast<uint8_t>(N - 1)};

    for(uint8_t i = 0; i < 4; i++) {
        // columns A, B, C, D
        place_queen(rows, cols, diags_u, diags_d, col_i[i], pre.raw[i], N);
        // rows E, F, G, H
        place_queen(rows, cols, diags_u, diags_d, pre.raw[i + 4], row_i[i], N);
    }
}

void go_up(uint_fast32_t& rows, uint_fast64_t& diags_u, uint_fast64_t& diags_d, uint8_t steps) {
    rows >>= steps;
    diags_d >>= steps;
    diags_u <<= steps;
}

void skip_preplaced(uint_fast32_t& rows, uint_fast64_t& diags_u, uint_fast64_t& diags_d) {
    if((0b1111 & rows) == 0b1111) {
        go_up(rows, diags_u, diags_d, 4);
    } else if((0b111 & rows) == 0b111) {
        go_up(rows, diags_u, diags_d, 3);
    } else if((0b11 & rows) == 0b11) {
        go_up(rows, diags_u, diags_d, 2);
    } else if((0b1 & rows) == 0b1) {
        go_up(rows, diags_u, diags_d, 1);
    }
}

uint64_t cpuSolver::solve_subboard(const std::vector<Preplacement> &starts)
{
    // counter for the number of solutions
    // sufficient until n=29
    uint_fast64_t num = 0;
    size_t start_cnt = starts.size();

    const uint64_t board_mask = (UINT32_C(1) << boardsize) - 1;

  #pragma omp parallel for reduction(+ : num) schedule(dynamic)
    for (size_t cnt = 0; cnt < start_cnt; cnt++) {
        uint_fast64_t l_num = 0;
        uint_fast32_t cols[MAXN] = {0}, rows[MAXN] = {0}, posibs[MAXN] = {0}; // Our backtracking 'stack'
        uint_fast64_t diagu[MAXN] = {0}, diagd[MAXN] = {0};
        int_fast16_t d = 1; // d is our depth in the backtrack stack
        // The UINT_FAST32_MAX here is used to fill all 'coloumn' bits after n ...

        load_preplacement(rows[d], cols[d], diagu[d], diagd[d], starts[cnt], boardsize);

        // check if preplacement is already full
        if(cols[d] == board_mask) {
            num++;
            //std::cout << "num: " << std::to_string(cnt) << " cnt:" << std::to_string(1) << std::endl;
            continue;
        }

        // E and F columns are always occupied, skip them
        if((0b11 & rows[d]) != 0b11) {
            std::cout << "Error, E and F not occupied" << std::endl;
        }
        go_up(rows[d], diagu[d], diagd[d], 2);

        skip_preplaced(rows[d], diagu[d], diagd[d]);

        // set unused high bits to 1, needed for possibility check
        cols[d] |= UINT_FAST32_MAX & ~board_mask;

        //  The variable posib contains the bitmask of possibilities we still have
        //  to try in a given row
        uint_fast32_t posib = ~(cols[d] | diagu[d] >> (boardsize - 1) | diagd[d]);

        while(d > 0) {
            while(posib) {
                uint_fast32_t bit = posib & (~posib + 1);
                posib ^= bit; // Eliminate the tried possibility.

                uint_fast64_t new_diagu = (diagu[d] | bit << (boardsize - 1)) << 1;
                uint_fast64_t new_diagd = (diagd[d] | bit) >> 1;
                uint_fast32_t new_cols = cols[d] | bit;
                uint_fast32_t new_rows = rows[d] >> 1;
                skip_preplaced(new_rows, new_diagu, new_diagd);
                uint_fast32_t new_posib = ~(new_cols | new_diagu  >> (boardsize - 1)| new_diagd);

                if(new_posib) {
                    posibs[d] = posib;

                    d += posib != 0; // avoid branching with this trick
                    posib = new_posib;

                    // make values current
                    rows[d] = new_rows;
                    cols[d] = new_cols;
                    diagu[d] = new_diagu;
                    diagd[d] = new_diagd;

                } else {
                    l_num += (new_cols == UINT_FAST32_MAX);
                }
            }
            d--;
            posib = posibs[d]; // backtrack ...
        }

        if(l_num) {
            //std::cout << "num: " << std::to_string(cnt) << " cnt:" << std::to_string(l_num) << std::endl;
        }
        num += l_num;
    }

    return num;
}
