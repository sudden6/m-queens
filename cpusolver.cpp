#include "cpusolver.h"
#include <iostream>
#include <cassert>
#include <chrono>


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

static constexpr uint_fast32_t switch_bit_mask_1 = 1 << 11;
static constexpr uint_fast32_t switch_bit_mask_2 = 1 << 5;

static uint_fast64_t stat_lookups = 0;
static uint_fast64_t stat_lookups_found = 0;
static uint_fast64_t stat_cmps = 0;
static uint_fast64_t stat_last_0 = 0;
static uint_fast64_t stat_last_1 = 0;
static uint_fast64_t stat_last_2 = 0;


uint64_t cpuSolver::get_solution_cnt(uint32_t cols, uint32_t diagl, uint32_t diagr) {
    uint64_t solutions = 0;
    const auto& found = lookup_hash.find(cols);
    // std::cout << std::hex << (bit & 0xFF) << std::endl;
    if(found != lookup_hash.end()) {
        //stat_lookups_found++;
        uint64_t search_elem = (static_cast<uint64_t>(diagr) << 32) | diagl;

        std::array<uint64_t, lut_vec_size>& new_vec =  found->second;
        // comparison loop

        const size_t candidates = new_vec.size();
        //stat_cmps += candidates;

        //stat_last_0++;

        #pragma omp simd //reduction(+ : solutions)
        for(size_t i = 0; i < candidates; i++) {
            solutions += (new_vec[i] & search_elem) == 0;
        }
    }

    return solutions;
}

void update_bit_stats(std::vector<uint_fast64_t>& state, uint64_t bits, size_t num_bits) {
    if (state.size() < num_bits) {
        return;
    }

    if (sizeof(uint64_t)*8 < num_bits) {
        return;
    }
    uint64_t mask = 1;
    for(size_t i = 0; i < num_bits; i++) {
        if(bits & mask) {
            state[i]++;
        }
        mask <<= 1;
    }
}

uint64_t cpuSolver::solve_subboard(const std::vector<start_condition>& starts) {

  std::vector<uint_fast64_t> stat_diagl;
  std::vector<uint_fast64_t> stat_diagr;

  stat_diagl.resize(boardsize, 0);
  stat_diagr.resize(boardsize, 0);
  stat_lookups = 0;
  stat_lookups_found = 0;
  stat_cmps = 0;
  stat_last_0 = 0;
  stat_last_1 = 0;
  stat_last_2 = 0;
  std::cout << "Solving N=" << std::to_string(boardsize) << std::endl;
  // counter for the number of solutions
  // sufficient until n=29
  uint_fast64_t num_lookup = 0;

  const size_t start_cnt = starts.size();  

  const uint_fast32_t board_width_mask = ~(UINT_FAST32_MAX << boardsize);
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

  const uint8_t lookup_depth = 3;

  auto lut_init_time_start = std::chrono::high_resolution_clock::now();
  init_lookup(lookup_depth, mask);
  auto lut_init_time_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = lut_init_time_end - lut_init_time_start;

  std::cout << "Time to init lookup table: " << std::to_string(elapsed.count()) << "s" << std::endl;

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
                //stat_lookups++;
                //update_bit_stats(stat_diagl, board_width_mask & new_diagl, boardsize);
                //update_bit_stats(stat_diagr, board_width_mask & new_diagr, boardsize);

                num_lookup += get_solution_cnt(bit, new_diagl, new_diagr);
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
        }
      }
      d--;
      posib = posibs[d]; // backtrack ...
    }
  }

  std::cout << "Lookups: " << std::to_string(stat_lookups) << std::endl;
  std::cout << "Lookups found: " << std::to_string(stat_lookups_found) << std::endl;
  std::cout << "Compares: " << std::to_string(stat_cmps) << std::endl;
  std::cout << "Last 0: " << std::to_string(stat_last_0) << std::endl;
  std::cout << "Last 1: " << std::to_string(stat_last_1) << std::endl;
  std::cout << "Last 2: " << std::to_string(stat_last_2) << std::endl;
  std::cout << "Solutions Lookup: " << std::to_string(num_lookup*2) << std::endl;

  return num_lookup * 2;
}

size_t cpuSolver::init_lookup(uint8_t depth, uint32_t skip_mask)
{
    std::cout << "Building Lookup Table" << std::endl;
    std::cout << "LUT depth: " << std::to_string(depth) << std::endl;

    uint_fast64_t stat_total = 0;
    std::vector<uint_fast64_t> stat_diagl;
    std::vector<uint_fast64_t> stat_diagr;

    stat_diagl.resize(boardsize, 0);
    stat_diagr.resize(boardsize, 0);

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
      diagr[d] = bit0 << (depth - 1);
      // we're allready two rows into the field here
      // TODO: find out why -2 is needed here
      rest[d] = static_cast<int8_t>(depth - 2);

      //  The variable posib contains the bitmask of possibilities we still have
      //  to try in a given row ...
      uint_fast32_t posib = (cols[d] | diagl[d] | (diagr[d] >> depth));

      diagl[d] <<= 1;
      diagr[d] >>= 1;

      while (d > 0) {
        int8_t l_rest = rest[d];

        while (posib != UINT_FAST32_MAX) {
          // The standard trick for getting the rightmost bit in the mask
          uint_fast32_t bit = ~posib & (posib + 1);
          posib ^= bit; // Eliminate the tried possibility.
          uint_fast32_t new_diagl = (bit << 1) | diagl[d];
          uint_fast32_t new_diagr = (bit << (depth - 1)) | diagr[d];
          bit |= cols[d];
          uint_fast32_t new_posib = (bit | new_diagl | (new_diagr >> depth));

          if (new_posib != UINT_FAST32_MAX) {
              if (bit & skip_mask) {
                  continue;
              }

              if (l_rest == 0) {
                  uint_fast32_t conv = ~bit | (UINT_FAST32_MAX << boardsize);
                  const uint_fast32_t board_width_mask = ~(UINT_FAST32_MAX << boardsize);
                  uint_fast32_t conv_diagl = (new_diagl >> depth) & board_width_mask;

                  // TODO(sudden6): pulling bits out of thin air, need to fix
                  uint_fast32_t conv_diagr = new_diagr & board_width_mask;
                  // std::cout << std::hex << (conv & 0xFF) << std::endl;

                  if (__builtin_popcount(conv_diagl) != depth) {
                      //std::cout << "Info lost in diagl" << std::endl;
                  }
                  if (__builtin_popcount(conv_diagr) != depth) {
                      //std::cout << "Info lost in diagr" << std::endl;
                  }

                  stat_total++;
                  update_bit_stats(stat_diagl, conv_diagl, boardsize);
                  update_bit_stats(stat_diagr, conv_diagr, boardsize);

                  uint64_t new_entry = (static_cast<uint64_t>(conv_diagr) << 32) | static_cast<uint32_t>(conv_diagl);

                  auto it = lookup_hash.find(conv);
                  if (it == lookup_hash.end()) {
                      std::array<uint64_t, lut_vec_size> new_vec = {new_entry};
                      for(size_t i = 1; i < new_vec.size(); i++) {
                          new_vec[i] = UINT64_MAX;
                      }

                      lookup_hash.emplace(conv, new_vec);

                      num++;
                  } else {
                      auto& vec = it->second;

                      for(size_t i = 0; i < vec.size(); i++) {
                          if(vec[i] == UINT64_MAX) {
                            vec[i] = new_entry;
                            break;
                          }
                      }
                      num++;
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
    uint_fast64_t max_len_0 = 0;
    uint64_t max_cnt = 0;
    for(const auto& vec : lookup_hash) {
        uint_fast64_t cnt = 0;
        for(size_t i = 0; i < vec.second.size(); i++) {
            if(vec.second[i] == UINT64_MAX) {
                break;
            }
            cnt++;
        }
        max_len_0 = std::max(cnt, max_len_0);
    }
    std::cout << "Max Vec len: " << std::to_string(max_len_0) << std::endl;
    std::cout << "Max Elem cnt: " << std::to_string(max_cnt) << std::endl;
    std::cout << "Unsolvable: " << std::to_string(unsolvable) << std::endl;
    std::cout << "Total: " << std::to_string(stat_total) << std::endl;

    return num;
}
