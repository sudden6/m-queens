# m-queens

Multi-queens, a very fast solver for the general
[eight queens problem](https://en.wikipedia.org/wiki/Eight_queens_puzzle) with OpenMP support.

Based on code from https://rosettacode.org/wiki/N-queens_problem#C but much more optimized.

If you find a way to make the code even faster, please let me know.
Also feel free to open a Pull Request to add your own benchmarks.

# Benchmarks
## Single-Thread

Compiled with `gcc-7 -o m-queens.bin main.c -O2 -march=native -mtune=native -std=c99` and
run with `./m-queens 18`.

|CPU|N|Time to solve|
|---|---|---|
|Intel(R) Core(TM) i7-4600M CPU@ 3.5 GHz (single core)|18|129.56s|

## Multi-Thread

Compiled with `gcc-7 -o m-queens.bin main.c -O2 -march=native -mtune=native -std=c99 -fopenmp` and
run with `./m-queens 18`.

|CPU|Cores (real/virtual)|N|Time to solve|Speedup from Single-Thread|
|---|---|---|---|---|
|Intel(R) Core(TM) i7-4600M CPU@ 3.4 GHz (all cores)|2/4|18|52.56s|2.46|

# Future Improvements

### Use the optimized placement mentioned in https://github.com/preusser/q27 to reduce symmetries.
probably very hard to do with the current implementation

### Use OpenCL to make use of GPUs
Work in progress in the `sieves`, `split2_debug` and `ringbuffer` branches.
