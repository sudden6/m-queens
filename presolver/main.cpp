#include "cxxopts.hpp"
#include "presolver.h"
#include "start_file.h"

#include <string>

using start_con_vec = std::vector<start_condition_t>;

class Writer {
    public:
        Writer(uint8_t boardsize, uint8_t depth) {
            this->boardsize = boardsize;
            this->depth = depth;
        }

        void write_file(const start_con_vec& starts, size_t start_idx, size_t end_idx) {
            const std::string filename = "N_" + std::to_string(boardsize)
                                       + "_D_" + std::to_string(depth)
                                       + "_" + std::to_string(start_idx)
                                       + "_" + std::to_string(end_idx)
                                       + ".pre";

            if(!start_file::save_all(starts, filename)) {
                exit(EXIT_FAILURE);
            }
        }
    private:
        uint8_t boardsize = 0;
        uint8_t depth = 0;
};

int main(int argc, char **argv) {
    uint8_t boardsize = 0;
    uint8_t depth = 0;
    size_t chunk_size = 1000*1000;
    bool help = false;
    try
    {
      cxxopts::Options options("presolver", " - a workunit gernerator for m-queens2");
      options.add_options()
        ("b,boardsize", "[5..29] size of the board", cxxopts::value(boardsize))
        ("d,depth", "[3..(boardsize-1)] number of rows to presolver", cxxopts::value(depth))
        ("c,chunksize", "(default=1M) Number of start conditions to store in one file", cxxopts::value(chunk_size))
        ("h,help", "Print this information", cxxopts::value(help))
        ;

      auto result = options.parse(argc, argv);
      if (result.arguments().size() == 0) {
          help = true;
      }

      if (help)
      {
        std::cout << options.help();
        exit(EXIT_SUCCESS);
      }

      if (boardsize < 5 || boardsize > 29)
      {
        std::cout << "boardsize must be in the range [5..29]" << std::endl;
        exit(1);
      }

      if (depth >= boardsize || depth < 3)
      {
        std::cout << "depth must be in the range [3..(boardsize-1)]" << std::endl;
        exit(EXIT_FAILURE);
      }

      if (chunk_size == 0)
      {
          std::cout << "chunksize must be at least 1" << std::endl;
          exit(EXIT_FAILURE);
      }

    } catch (const cxxopts::OptionException& e)
    {
      std::cout << "error parsing options: " << e.what() << std::endl;
      exit(EXIT_FAILURE);
    }

    start_con_vec start = PreSolver::create_preplacement(boardsize);
    Writer w{boardsize, depth};

    if(depth == 3) {
        for(size_t i = 0; i < start.size(); i += chunk_size) {
            start_con_vec write_buf;
            write_buf.reserve(chunk_size);
            size_t end = std::min(i + chunk_size, start.size());
            for(size_t j = i; j < end; j++) {
                write_buf.push_back(start[j]);
            }

            w.write_file(write_buf, i, end - 1);
        }

        exit(EXIT_SUCCESS);
    }

    size_t pre_idx = 0;
    auto pre = PreSolver(boardsize, 3, depth - 3, start[pre_idx]);
    start_con_vec write_buf;
    write_buf.resize(chunk_size);
    auto curIt = write_buf.begin();

    size_t write_idx = 0;
    size_t end_idx = 0;
    while(pre_idx < start.size()) {
        while(curIt < write_buf.cend()) {
            curIt = pre.getNext(curIt, write_buf.cend());
            if(!pre.empty()) {
                continue;
            }
            pre_idx++;
            if(pre_idx == start.size()) {
                break;
            }
            pre = PreSolver(boardsize, 3, depth - 3, start[pre_idx]);
            if(pre.empty()) {
                break;
            }
        }
        const size_t write_size = std::distance(write_buf.begin(), curIt);
        end_idx = write_idx + write_size;
        write_buf.resize(write_size);
        w.write_file(write_buf, write_idx, end_idx - 1);
        write_buf.resize(chunk_size);
        write_idx = end_idx;
        curIt = write_buf.begin();
    }

    std::cout << "Wrote " << std::to_string(end_idx) << " records" << std::endl;

    exit(EXIT_SUCCESS);
}
