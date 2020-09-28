#include "cpusolver.h"
#include <iostream>
#include <cassert>
#include <chrono>
#include <omp.h>
#include <numeric>

cpuSolver::cpuSolver()
{
}

constexpr uint_fast8_t MINN = 6;
constexpr uint_fast8_t MAXN = 29;

#define STATS 0

uint8_t cpuSolver::lookup_depth(uint8_t boardsize, uint8_t placed)
{
    assert(boardsize > placed);
    uint8_t available_depth = boardsize - placed;
    available_depth -= 1; // main solver needs at least 1 queen to work with

    uint8_t limit = 5;

    return std::min(available_depth, limit);
}

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

uint64_t cpuSolver::count_solutions(const aligned_vec<diags_packed_t>& solutions, const aligned_vec<diags_packed_t>& candidates) {
    uint32_t solutions_cnt = 0;
    const size_t sol_size = solutions.size();
    const size_t can_size = candidates.size();
    const diags_packed_t* __restrict__ sol_data = solutions.data();
    const diags_packed_t* __restrict__ can_data = candidates.data();
#if STATS == 1
    stat_cmps += sol_size * can_size;
#endif

    for(size_t c_idx = 0; c_idx < can_size; c_idx++) {
#if STATS == 1
        stat_solver_diagl.update(can_data[c_idx].diagl);
        stat_solver_diagr.update(can_data[c_idx].diagr);
#endif
#pragma omp simd reduction(+:solutions_cnt) aligned(sol_data, can_data:32)
        for(size_t s_idx = 0; s_idx < sol_size; s_idx++) {
            solutions_cnt += (sol_data[s_idx].diagr & can_data[c_idx].diagr) == 0 && (sol_data[s_idx].diagl & can_data[c_idx].diagl) == 0;
        }
    }

    return solutions_cnt;
}

__attribute__ ((target_clones("default", "sse2", "sse4.2", "avx2")))
static uint32_t count_solutions_fixed(size_t batch,
                                          const aligned_vec<diags_packed_t>& solutions,
                                          const aligned_vec<diags_packed_t>& candidates) {
    const size_t sol_size = solutions.size();
    assert(sol_size % 2 == 0);
    const diags_packed_t* __restrict__ sol_data = solutions.data();
    const diags_packed_t* __restrict__ can_data = candidates.data();

    // prevent overflow in result variable
    assert(sol_size * batch < UINT32_MAX);
    uint32_t solutions_cnt = static_cast<uint32_t>(sol_size * batch);
    for(size_t s_idx = 0; s_idx < sol_size; s_idx++) {
#pragma omp simd reduction(-:solutions_cnt) aligned(sol_data, can_data: AVX2_alignment)
        for(size_t c_idx = 0; c_idx < batch; c_idx++) {
            solutions_cnt -= (sol_data[s_idx].diagr & can_data[c_idx].diagr) || (sol_data[s_idx].diagl & can_data[c_idx].diagl);
        }
    }

    return solutions_cnt;
}

uint64_t cpuSolver::get_solution_cnt(uint32_t cols, diags_packed_t search_elem, lut_t &lookup_candidates_high_prob, lut_t &lookup_candidates_low_prob) {
    uint64_t solutions_cnt = 0;

    // we only work with perfect lookup tables that have all possible column combinations,
    // so this will always return an element
    const auto& found = lookup_hash.find(cols);
    assert(found != lookup_hash.end());

#if STATS == 1
    stat_solver_diagl.update(search_elem.diagl);
    stat_solver_diagr.update(search_elem.diagr);
#endif

    auto lookup_idx = found->second;
    const bool high_prob = prob_mask & search_elem.diagr;

    auto& candidates_vec = high_prob ? lookup_candidates_high_prob[lookup_idx] : lookup_candidates_low_prob[lookup_idx];

    candidates_vec.push_back(search_elem);

    if(candidates_vec.size() == max_candidates) {
        solutions_cnt = count_solutions_fixed(max_candidates, lookup_solutions_low_prob[lookup_idx], candidates_vec);
#if STATS == 1
    stat_cmps += max_candidates * lookup_solutions_low_prob[lookup_idx].size();
#endif
        if (!high_prob) {
            solutions_cnt += count_solutions_fixed(max_candidates, lookup_solutions_high_prob[lookup_idx], candidates_vec);
#if STATS == 1
    stat_cmps += max_candidates * lookup_solutions_high_prob[lookup_idx].size();
#endif
        } else {
#if STATS == 1
            stat_high_prob += max_candidates;
#endif
        }
        candidates_vec.clear();
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

uint64_t cpuSolver::solve_subboard(const std::vector<start_condition_t> &starts) {
#if STATS == 1
  stat_lookups = 0;
  stat_cmps = 0;
  stat_high_prob = 0;
  stat_prob_saved = 0;

  stat_solver_diagl = bit_probabilities(boardsize);
  stat_solver_diagr = bit_probabilities(boardsize);
#endif

  std::cout << "Solving N=" << std::to_string(boardsize) << std::endl;
  // counter for the number of solutions
  // sufficient until n=29
  uint_fast64_t num_lookup = 0;

  const size_t start_cnt = starts.size();  

  uint32_t col_mask = UINT32_MAX;

  for (const auto& start : starts) {
      col_mask &= start.cols;
  }

  // clear highest padding bits
  col_mask &= ~(UINT32_MAX << boardsize);

  // count fixed bits overall
  uint8_t mask = 0;
  for (uint8_t i = 0; i < 32; i++) {
      if ((UINT32_C(1) << i) & col_mask) {
          mask++;
      }
  }

  std::cout << "Column mask: " << std::hex << col_mask << " zeros in mask: " << std::to_string(mask) << std::endl;

  lookup_hash.clear();
  lookup_solutions_low_prob.clear();
  lookup_solutions_high_prob.clear();

  uint8_t lut_depth = lookup_depth(boardsize, placed);
  auto lut_init_time_start = std::chrono::high_resolution_clock::now();
  size_t lut_size = init_lookup(lut_depth, col_mask);
  auto lut_init_time_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = lut_init_time_end - lut_init_time_start;

  std::cout << "Time to init lookup table: " << std::to_string(elapsed.count()) << "s" << std::endl;
  const bool perfect_lut = is_perfect_lut(lut_depth, boardsize - mask, lut_size);
  if(!perfect_lut) {
      std::cout << "Imperfect LUT, can't  work with that" << std::endl;
      return 0;
  }

  if (lut_size == 0) {
      std::cout << "Empty lookup table, can't work with that" << std::endl;
      return 0;
  }

  assert(this->boardsize > this->placed);

  constexpr uint8_t LOOKAHEAD = 3;
  const int8_t rest_init = static_cast<int8_t>(boardsize - LOOKAHEAD - this->placed);
  // TODO: find out why +1 is needed
  const int8_t rest_lookup = rest_init - (boardsize - this->placed - lut_depth) + 1;

#if STATS == 1
  const size_t omp_threads = 1;
#else
  const size_t omp_threads = omp_get_max_threads();
#endif

  const size_t thread_cnt = std::min(omp_threads, start_cnt);

  // first index: lookup table index
  std::vector<lut_t> thread_luts_high_prob;
  std::vector<lut_t> thread_luts_low_prob;

  for(size_t t = 0; t < thread_cnt; t++) {
      lut_t new_lut_high_prob;
      lut_t new_lut_low_prob;

      new_lut_high_prob.reserve(lookup_solutions_low_prob.size());
      new_lut_low_prob.reserve(lookup_solutions_low_prob.size());
      for (size_t i = 0; i < lookup_solutions_low_prob.size(); i++) {
          new_lut_high_prob.emplace_back(aligned_vec<diags_packed_t>(max_candidates));
      }
      for (size_t i = 0; i < lookup_solutions_low_prob.size(); i++) {
          new_lut_low_prob.emplace_back(aligned_vec<diags_packed_t>(max_candidates));
      }

      thread_luts_high_prob.push_back(std::move(new_lut_high_prob));
      thread_luts_low_prob.push_back(std::move(new_lut_low_prob));

  }

#pragma omp parallel for reduction(+ : num_lookup) num_threads(thread_cnt) schedule(dynamic)
  for (uint_fast32_t cnt = 0; cnt < start_cnt; cnt++) {
    const size_t thread_num = static_cast<size_t>(omp_get_thread_num());
    lut_t& lookup_candidates_high_prob = thread_luts_high_prob[thread_num];
    lut_t& lookup_candidates_low_prob = thread_luts_low_prob[thread_num];

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
            uint_fast32_t allowed2 = l_rest > static_cast<int8_t>(0);

            if(allowed2 && ((lookahead2 == UINT_FAST32_MAX) || (lookahead1 == UINT_FAST32_MAX))) {
                continue;
            }

#if 0
            if(allowed2 && (lookahead1 == lookahead2)) {
                uint_fast32_t adjacent = ~lookahead1 & (~lookahead1 << 1);
                if ((adjacent != 0) && (__builtin_popcount(~lookahead1) == 2)) {
                    continue;
                }

            }
#endif

            if (l_rest == rest_lookup) {
                // compute final lookup_depth stages via hashtable lookup
                diags_packed_t candidate = {static_cast<uint32_t>(new_diagr), static_cast<uint32_t>(new_diagl)};
#if STATS == 1
                stat_lookups++;
                stat_solver_diagl.update(candidate.diagl);
                stat_solver_diagr.update(candidate.diagr);
#endif
                num_lookup += get_solution_cnt(static_cast<uint32_t>(bit), candidate, lookup_candidates_high_prob, lookup_candidates_low_prob);
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
  auto lut_clean_time_start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for reduction(+ : num_lookup) num_threads(thread_cnt) schedule(dynamic)
  for(size_t t = 0; t < thread_luts_high_prob.size(); t++) {
      for (auto& it : lookup_hash) {
          auto lookup_idx = it.second;
          if(thread_luts_high_prob[t][lookup_idx].size() > 0) {
              num_lookup += count_solutions(lookup_solutions_low_prob[lookup_idx], thread_luts_high_prob[t][lookup_idx]);
#if STATS == 1
              stat_high_prob += thread_luts_high_prob[t][lookup_idx].size();
#endif
          }
          if(thread_luts_low_prob[t][lookup_idx].size() > 0) {
              num_lookup += count_solutions(lookup_solutions_low_prob[lookup_idx], thread_luts_low_prob[t][lookup_idx]);
              num_lookup += count_solutions(lookup_solutions_high_prob[lookup_idx], thread_luts_low_prob[t][lookup_idx]);
          }
      }
  }

#if STATS == 1
  auto diagl_probs = stat_solver_diagl.get_probablities();
  auto diagr_probs = stat_solver_diagr.get_probablities();
#endif

  auto lut_clean_time_end = std::chrono::high_resolution_clock::now();
  elapsed = lut_clean_time_end - lut_clean_time_start;

  std::cout << "Time to clean lookup table: " << std::to_string(elapsed.count()) << "s" << std::endl;

#if STATS == 1
  std::cout << "Lookups: " << std::to_string(stat_lookups) << std::endl;
  std::cout << "Compares: " << std::to_string(stat_cmps) << std::endl;
  std::cout << "High Prob hits: " << std::to_string(stat_high_prob) << std::endl;
  std::cout << "High Prob hit factor " << std::to_string(stat_high_prob/static_cast<float>(stat_lookups)) << std::endl;
#endif

  return num_lookup * 2;
}


__uint128_t cpuSolver::factorial(uint8_t n)
{
    // prevent the result from overflowing
    assert(n <= 34);
    if(n > 1) {
        return n * factorial(n - 1);
    } else {
        return 1;
    }
}

bool cpuSolver::is_perfect_lut(uint8_t lut_depth, uint8_t free_bits, uint64_t entries)
{
    uint64_t max = factorial(free_bits) / (factorial(lut_depth) * factorial(free_bits - lut_depth));
    return max == entries;
}

size_t cpuSolver::init_lookup(uint8_t depth, uint32_t skip_mask)
{
    std::cout << "Building Lookup Table" << std::endl;
    std::cout << "LUT depth: " << std::to_string(depth) << std::endl;

    if (depth < 2) {
        std::cout << "ERROR: Minimum depth for lookup table is 2" << std::endl;
        return 0;
    }

    uint_fast64_t stat_total = 0;

    // stat counter for number of elements in the lookup table
    uint_fast64_t num = 0;
    // stat counter of how many bit patterns have more than one entry
    uint_fast64_t multiple_entries = 0;
    // stat counter for non solvable combinations
    uint_fast64_t unsolvable = 0;

    // preliminary lookup table, not yet aligned
    // first dimension indexed by indices in lookup_hash
    std::vector<std::vector<diags_packed_t>> lookup_proto;

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

                // In the solver we look for free columns, so we must invert the bits here
                // fill up with ones to match padding in the solver
                uint_fast32_t conv = ~bit | (UINT_FAST32_MAX << boardsize);
                const uint_fast32_t board_width_mask = ~(UINT_FAST32_MAX << boardsize);
                // diagonals are not inverted, they have to be compared in the solver step
                // we loose some bits here, but they do not matter for a solution
                uint_fast32_t conv_diagl = (new_diagl >> depth) & board_width_mask;
                uint_fast32_t conv_diagr = new_diagr & board_width_mask;

                if (bit & skip_mask) {
                    continue;
                }

                // check if at the correct depth to save to the lookup table
                if (l_rest == 0) {
                    stat_total++;

                    // combine diagonals for easier handling
                    diags_packed_t new_entry = {static_cast<uint32_t>(conv_diagr), static_cast<uint32_t>(conv_diagl)};

                    auto it = lookup_hash.find(static_cast<uint32_t>(conv));

                    if (it != lookup_hash.end()) {
                        auto pattern_idx = it->second;
                        lookup_proto[pattern_idx].push_back(new_entry);
                        num++;
                    } else {
                        // column pattern doesn't yet exist, create one
                        lookup_hash.emplace(conv, lookup_proto.size());
                        // create element in the lookup table too
                        std::vector<diags_packed_t> new_vec{new_entry};
                        lookup_proto.push_back(new_vec);

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

            }
            d--;
            posib = posibs[d]; // backtrack ...
        }
    }

    bit_probabilities stat_total_probs_l(boardsize);
    bit_probabilities stat_total_probs_r(boardsize);

    size_t stat_max_len = 0;
    size_t stat_min_len = SIZE_MAX;
    size_t stat_final_lut_size = 0;
    size_t stat_high_prob_cnt = 0;
    size_t stat_low_prob_cnt = 0;

    // This seems to be a sweetspot, more research on the probabilities needed
    prob_mask = UINT32_C(1) << (depth - 1);

    // put lookup table prototype in final aligned form
    for(size_t i = 0; i < lookup_proto.size(); i++) {
        auto& lookup_vec = lookup_proto[i];
        size_t elemen_cnt = lookup_vec.size();
        stat_max_len = std::max(stat_max_len, elemen_cnt);
        stat_min_len = std::min(stat_min_len, elemen_cnt);

#if 0
        for(const auto& elem: lookup_vec) {
            stat_total_probs_l.update(elem.diagl);
            stat_total_probs_r.update(elem.diagr);
        }

#endif

        // prob_mask bits set
        std::vector<diags_packed_t> high_prob;
        // prob_mask bits not set
        std::vector<diags_packed_t> low_prob;

        for(const auto& diag: lookup_vec) {
            if(diag.diagr & prob_mask) {
                high_prob.push_back(diag);
                stat_high_prob_cnt++;
            } else {
                low_prob.push_back(diag);
                stat_low_prob_cnt++;
            }
        }

        constexpr diags_packed_t dummy_element {UINT32_MAX, UINT32_MAX};

        // check if elements are sufficiently alligned
        if (high_prob.size() % 2 != 0) {
            high_prob.push_back(dummy_element);
        }

        if (low_prob.size() % 2 != 0) {
            low_prob.push_back(dummy_element);
        }

        // put in final lookup table
        aligned_vec<diags_packed_t> new_vec_single(low_prob.size(), low_prob.size());
        std::memcpy(new_vec_single.data(), low_prob.data(), low_prob.size() * sizeof (diags_packed_t));
        lookup_solutions_low_prob.push_back(std::move(new_vec_single));

        aligned_vec<diags_packed_t> new_vec_high_prob(high_prob.size(), high_prob.size());
        std::memcpy(new_vec_high_prob.data(), high_prob.data(), high_prob.size() * sizeof (diags_packed_t));
        lookup_solutions_high_prob.push_back(std::move(new_vec_high_prob));
    }

    auto probs_l = stat_total_probs_l.get_probablities();
    auto probs_r = stat_total_probs_r.get_probablities();

    if (stat_max_len * max_candidates > UINT32_MAX) {
        std::cout << "ERROR: Possible overflow in count_solutions_fixed(...)" << std::endl;
        return 0;
    }

    std::cout << "Hashtable keys: " << std::to_string(lookup_hash.size()) << std::endl;
    std::cout << "Lookup entries: " << std::to_string(num) << std::endl;
    std::cout << "Lookup final size: " << std::to_string(stat_final_lut_size) << std::endl;
    std::cout << "Multiple entries: " << std::to_string(multiple_entries) << std::endl;
    std::cout << "Avg Vec len: " << std::to_string(static_cast<double>(num)/lookup_hash.size()) << std::endl;
    std::cout << "Max Vec len: " << std::to_string(stat_max_len) << std::endl;
    std::cout << "Min Vec len: " << std::to_string(stat_min_len) << std::endl;
    std::cout << "Unsolvable: " << std::to_string(unsolvable) << std::endl;
    std::cout << "Total: " << std::to_string(stat_total) << std::endl;
    std::cout << "High/Low prob ratio: " << std::to_string(stat_high_prob_cnt/static_cast<float>(stat_low_prob_cnt)) << std::endl;

    return lookup_hash.size();
}
