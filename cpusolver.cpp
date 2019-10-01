#include "cpusolver.h"
#include <iostream>
#include <cassert>
#include <chrono>

#include <emmintrin.h>
#include <xmmintrin.h>

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

uint64_t cpuSolver::count_solutions(const aligned_ABvec<diags_packed_t, lut_vec_size, max_candidates> &candidates) {
    uint32_t solutions_cnt = 0;
    const size_t sol_size = candidates.sizeA();
    const size_t can_size = candidates.sizeB();
    const diags_packed_t* __restrict__ sol_data = candidates.dataA();
    const diags_packed_t* __restrict__ can_data = candidates.dataB();

    for(size_t c_idx = 0; c_idx < can_size; c_idx++) {
#pragma omp simd reduction(+:solutions_cnt)
        for(size_t s_idx = 0; s_idx < sol_size; s_idx++) {
            solutions_cnt += (sol_data[s_idx].diagr & can_data[c_idx].diagr) == 0 && (sol_data[s_idx].diagl & can_data[c_idx].diagl) == 0;
        }
    }

    return solutions_cnt;
}

uint32_t cpuSolver::count_solutions_fixed(const aligned_ABvec<diags_packed_t, lut_vec_size, max_candidates>& candidates) {
    const size_t sol_size = candidates.sizeA();
    const diags_packed_t* __restrict__ sol_data = candidates.dataA();
    const diags_packed_t* __restrict__ can_data = candidates.dataB();

    /*
    for(size_t s_idx = 0; s_idx < sol_size; s_idx++) {
#pragma omp simd reduction(+:solutions_cnt)
        for(size_t c_idx = 0; c_idx < max_candidates; c_idx++) {
            solutions_cnt += (sol_data[s_idx].diagr & can_data[c_idx].diagr) == 0 && (sol_data[s_idx].diagl & can_data[c_idx].diagl) == 0;
        }
    }
    */

    /*
    uint32_t solutions_cnt = sol_size * max_candidates;
    for(size_t s_idx = 0; s_idx < sol_size; s_idx++) {
#pragma omp simd reduction(+:solutions_cnt)
        for(size_t c_idx = 0; c_idx < max_candidates; c_idx++) {
            solutions_cnt -= (sol_data[s_idx].diagr & can_data[c_idx].diagr) || (sol_data[s_idx].diagl & can_data[c_idx].diagl);
        }
    }
    */

    __m128i sol_cnt_0_1_2_3 = _mm_setzero_si128();
    __m128i zero_vec = _mm_setzero_si128();

    for(size_t s_idx = 0; s_idx < sol_size; s_idx += 2) {
        __m128i sol_data_0_1 = _mm_load_si128((const __m128i*) &sol_data[s_idx]);
        __m128i sol_data_1_0 = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(sol_data_0_1), _mm_castsi128_pd(sol_data_0_1), 1));

        for(size_t c_idx = 0; c_idx < max_candidates; c_idx += 2) {
            __m128i can_data_0_1 = _mm_load_si128((const __m128i*) &can_data[c_idx]);
            __m128i and_res_0_1 = _mm_and_si128(sol_data_0_1, can_data_0_1);
            __m128i and_res_2_3 = _mm_and_si128(sol_data_1_0, can_data_0_1);

            __m128 and_res_0_1_2_3r = _mm_shuffle_ps(_mm_castsi128_ps(and_res_0_1), _mm_castsi128_ps(and_res_2_3), _MM_SHUFFLE(2, 0, 2, 0));
            __m128 and_res_0_1_2_3l = _mm_shuffle_ps(_mm_castsi128_ps(and_res_0_1), _mm_castsi128_ps(and_res_2_3), _MM_SHUFFLE(3, 1, 3, 1));

            __m128i or_res = _mm_or_si128(_mm_castps_si128(and_res_0_1_2_3r), _mm_castps_si128(and_res_0_1_2_3l));
            __m128i cmp_res = _mm_cmpgt_epi32(or_res, zero_vec);

            sol_cnt_0_1_2_3 = _mm_add_epi32(cmp_res, sol_cnt_0_1_2_3);
        }
    }

    uint32_t sol_cnt_vec[4];

    _mm_store_si128((__m128i*)sol_cnt_vec, sol_cnt_0_1_2_3);

    uint32_t sol_sum = sol_cnt_vec[0] + sol_cnt_vec[1] + sol_cnt_vec[2] + sol_cnt_vec[3];

    return sol_size * max_candidates + sol_sum;
}


uint64_t cpuSolver::get_solution_cnt(uint32_t cols, diags_packed_t search_elem) {
    uint64_t solutions_cnt = 0;
    const auto& found = lookup_hash.find(cols);

    // since the lookup table contains all possible combinations, we know the current one will be found
    assert(found != lookup_hash.end());

    auto& candidates_vec = found->second;
    candidates_vec.push_backB(search_elem);

    if(candidates_vec.sizeB() == max_candidates) {
        solutions_cnt = count_solutions_fixed(candidates_vec);
        candidates_vec.clearB();
    }

    return solutions_cnt;
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

  stat_lookups = 0;
  stat_lookups_found = 0;
  stat_cmps = 0;
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

  auto lut_init_time_start = std::chrono::high_resolution_clock::now();
  init_lookup(lookup_depth, mask);
  auto lut_init_time_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = lut_init_time_end - lut_init_time_start;

  std::cout << "Time to init lookup table: " << std::to_string(elapsed.count()) << "s" << std::endl;

#define LOOKAHEAD 3
  const int8_t rest_init = boardsize - LOOKAHEAD - this->placed;
  // TODO: find out why +1 is needed
  const int8_t rest_lookup = rest_init - (boardsize - this->placed - lookup_depth) + 1;

//#pragma omp parallel for firstprivate(lookup_hash) reduction(+ : num_lookup) schedule(dynamic)
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
                // compute final lookup_depth stages via hashtable lookup
                stat_lookups++;
                num_lookup += get_solution_cnt(bit, {.diagr = static_cast<uint32_t>(new_diagr), .diagl = static_cast<uint32_t>(new_diagl)});
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

  std::cout << "Cleaning Lookup table" << std::endl;

  for (auto& it : lookup_hash) {
      auto& candidates_vec = it.second;
      if(candidates_vec.sizeB() > 0) {
          num_lookup += count_solutions(candidates_vec);
      }
  }


  std::cout << "Lookups: " << std::to_string(stat_lookups) << std::endl;
  std::cout << "Lookups found: " << std::to_string(stat_lookups_found) << std::endl;
  std::cout << "Compares: " << std::to_string(stat_cmps) << std::endl;
  std::cout << "Solutions Lookup: " << std::to_string(num_lookup*2) << std::endl;

  return num_lookup * 2;
}

size_t cpuSolver::init_lookup(uint8_t depth, uint32_t skip_mask)
{
    std::cout << "Building Lookup Table" << std::endl;
    std::cout << "LUT depth: " << std::to_string(depth) << std::endl;
    aligned_ABvec<uint64_t, lut_vec_size, max_candidates> test;
    std::cout << "LUT element size: " << std::to_string(sizeof(test)) << std::endl;

    uint_fast64_t stat_total = 0;

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

              // check if at the correct depth to save to the lookup table
              if (l_rest == 0) {
                  // In the solver we look for free columns, so we must invert the bits here
                  // fill up with ones to match padding in the solver
                  uint32_t conv = ~bit | (UINT_FAST32_MAX << boardsize);
                  const uint_fast32_t board_width_mask = ~(UINT_FAST32_MAX << boardsize);
                  // diagonals are not inverted, they have to be compared in the solver step
                  // we loose some bits here, but they do not matter for a solution
                  uint_fast32_t conv_diagl = (new_diagl >> depth) & board_width_mask;
                  uint_fast32_t conv_diagr = new_diagr & board_width_mask;

                  stat_total++;

                  // combine diagonals for easier handling
                  diags_packed_t new_entry = {.diagr = static_cast<uint32_t>(conv_diagr), .diagl = static_cast<uint32_t>(conv_diagl)};

                  auto it = lookup_hash.find(conv);

                  if (it != lookup_hash.end()) {
                      auto& vec = it->second;
                      vec.push_backA(new_entry);
                      num++;
                  } else {
                      // column pattern doesn't yet exist, create one
                      aligned_ABvec<diags_packed_t, lut_vec_size, max_candidates> new_vec;
                      new_vec.push_backA(new_entry);
                      lookup_hash.emplace(conv, std::move(new_vec));

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
    size_t max_len = 0;
    size_t min_len = SIZE_MAX;
    size_t align_check = 0;
    uint64_t vec_idx = 0;
    for(const auto& vec : lookup_hash) {
        size_t vec_size = vec.second.sizeA();
        //std::cout << "Vec [" << std::to_string(vec_idx) << "] size: " << std::to_string(vec_size) << std::endl;
        vec_idx++;
        max_len = std::max(vec_size, max_len);
        min_len = std::min(vec_size, min_len);
        align_check |= vec_size;
    }
    std::cout << "Max Vec len: " << std::to_string(max_len) << std::endl;
    std::cout << "Min Vec len: " << std::to_string(max_len) << std::endl;
    std::cout << "Unsolvable: " << std::to_string(unsolvable) << std::endl;
    std::cout << "Total: " << std::to_string(stat_total) << std::endl;

    return num;
}
