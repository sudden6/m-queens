#include "cpusolver.h"
#include <iostream>

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

  uint_fast64_t stat_lookups = 0;
  uint_fast64_t stat_lookups_found = 0;
  std::cout << "Solving" << std::endl;
  // counter for the number of solutions
  // sufficient until n=29
  uint_fast64_t num = 0;
  const size_t start_cnt = starts.size();  

  uint32_t col_mask = UINT32_MAX;

  for (const auto& start : starts) {
      col_mask &= start.cols;
  }

  // count fixed bits overall
  uint8_t mask = 0;
  for (uint8_t i = 0; i < 32; i++) {
      if ((UINT32_C(1) << i) & col_mask) {
          mask++;
      }
  }

  // subtract overhead bits
  mask -= 32 - boardsize;

  std::cout << "Column mask: " << std::hex << col_mask << " zeros in mask: " << std::to_string(mask) << std::endl;

  lookup_hash.clear();
  const uint8_t lookup_depth = 7;
  init_lookup(lookup_depth, mask);

#define LOOKAHEAD 3
  const int8_t rest_init = boardsize - LOOKAHEAD - this->placed;
  // TODO: find out why +1 is needed
  const int8_t rest_lookup = rest_init - (boardsize - this->placed - lookup_depth) + 1;

//#pragma omp parallel for reduction(+ : num) schedule(dynamic)
  for (uint_fast32_t cnt = 0; cnt < start_cnt; cnt++) {
    uint_fast32_t cols[MAXN], posibs[MAXN]; // Our backtracking 'stack'
    uint_fast32_t diagl[MAXN], diagr[MAXN];
    int8_t rest[MAXN]; // number of rows left
    int_fast16_t d = 1; // d is our depth in the backtrack stack
    // The UINT_FAST32_MAX here is used to fill all 'coloumn' bits after n ...
    cols[d] = starts[cnt].cols | (UINT_FAST32_MAX << boardsize);
    // This places the first two queens
    diagl[d] = starts[cnt].diagl;
    diagr[d] = starts[cnt].diagr;
    // we're allready two rows into the field here
    rest[d] = rest_init;

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

            if (l_rest == rest_lookup) {
                stat_lookups++;

                const auto& found = lookup_hash.find(bit);
                // std::cout << std::hex << (bit & 0xFF) << std::endl;
                if(found != lookup_hash.end()) {
                    stat_lookups_found++;
                }
            }

          // The next two lines save stack depth + backtrack operations
          // when we passed the last possibility in a row.
          // Go lower in the stack, avoid branching by writing above the current
          // position
          posibs[d] = posib;
          d += posib != UINT_FAST32_MAX; // avoid branching with this trick
          posib = new_posib;

          l_rest--;

          // make values current
          cols[d] = bit;
          diagl[d] = new_diagl << 1;
          diagr[d] = new_diagr >> 1;
          rest[d] = l_rest;
        } else {
          // when all columns are used, we found a solution
          num += bit == UINT_FAST32_MAX;
        }
      }
      d--;
      posib = posibs[d]; // backtrack ...
    }
  }

  std::cout << "Lookups: " << std::to_string(stat_lookups) << std::endl;
  std::cout << "Lookups found: " << std::to_string(stat_lookups_found) << std::endl;


  return num * 2;
}

size_t cpuSolver::init_lookup(uint8_t depth, uint32_t skip_mask)
{
    std::cout << "Building Lookup Table" << std::endl;
    // stat counter for number of elements in the lookup table
    uint_fast64_t num = 0;
    // stat counter of how many bit patterns have more than one entry
    uint_fast64_t multiple_entries = 0;
    // stat counter for non solvable combinations
    uint_fast64_t unsolvable = 0;

    for (uint_fast8_t q0 = 0; q0 < boardsize; q0++) {
      uint_fast32_t bit0 = 1 << q0; // The first queen placed
      uint_fast32_t cols[MAXN], posibs[MAXN]; // Our backtracking 'stack'
      uint_fast32_t diagl[MAXN], diagr[MAXN];
      int8_t rest[MAXN]; // number of rows left
      int_fast16_t d = 1; // d is our depth in the backtrack stack
      // The UINT_FAST32_MAX here is used to fill all 'coloumn' bits after n ...
      cols[d] = bit0 | (UINT_FAST32_MAX << boardsize);
      // This places the first two queens
      diagl[d] = bit0 << 1;
      diagr[d] = bit0 >> 1;
      // we're allready two rows into the field here
      // TODO: find out why -2 is needed here
      rest[d] = static_cast<int8_t>(depth - 2);

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
              if (bit & skip_mask) {
                  continue;
              }

              if (l_rest == 0) {
                  uint_fast32_t conv = ~bit | (UINT_FAST32_MAX << boardsize);
                  // std::cout << std::hex << (conv & 0xFF) << std::endl;
                  auto it = lookup_hash.find(conv);
                  if (it == lookup_hash.end()) {
                      lookup_t new_entry = {.diag_r = new_diagr, .diag_l = new_diagl};
                      std::vector<lookup_t> new_vec = {new_entry};
                      lookup_hash.emplace(conv, new_vec);
                      num++;
                  } else {
                      auto& vec = it->second;

                      lookup_t new_entry = {.diag_r = new_diagr, .diag_l = new_diagl};
                      vec.push_back(new_entry);
                      num++;
                      multiple_entries++;
                  }
                  continue;
              }

            // The next two lines save stack depth + backtrack operations
            // when we passed the last possibility in a row.
            // Go lower in the stack, avoid branching by writing above the current
            // position
            posibs[d] = posib;
            d += posib != UINT_FAST32_MAX; // avoid branching with this trick
            posib = new_posib;

            l_rest--;

            // make values current
            cols[d] = bit;
            diagl[d] = new_diagl << 1;
            diagr[d] = new_diagr >> 1;
            rest[d] = l_rest;
          } else {
            // reached a point where we can't place more queens
            unsolvable++;
          }
        }
        d--;
        posib = posibs[d]; // backtrack ...
      }
    }

    std::cout << "Hashtable keys: " << std::to_string(lookup_hash.size()) << std::endl;
    std::cout << "Lookup entries: " << std::to_string(num) << std::endl;
    std::cout << "Multiple entries: " << std::to_string(multiple_entries) << std::endl;
    std::cout << "Avg Vec len: " << std::to_string(static_cast<double>(num)/lookup_hash.size()) << std::endl;
    uint_fast64_t max_len = 0;
    for(const auto& vec : lookup_hash) {
        max_len = std::max(vec.second.size(), max_len);
    }
    std::cout << "Max Vec len: " << std::to_string(max_len) << std::endl;
    std::cout << "Unsolvable: " << std::to_string(unsolvable) << std::endl;

    return num;
}
