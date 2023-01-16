# m-queens2
Very fast solver for the [n queens problem][1] with OpenMP and OpenCL support. It counts the number of solutions to the problem for a given board size.

Based on code from https://rosettacode.org/wiki/N-queens_problem#C but much more optimized.

If you find a way to make the code even faster, please let me know.
Also feel free to open a Pull Request to add your own benchmarks.

# Benchmarks
## Single-Thread

Command: `OMP_NUM_THREADS=1 ./m-queens2 -m cpu -s 18`

|CPU|N|Time to solve|Solutions per sec|
|---|---|---|---|
| AMD Ryzen 5 2400G | 18 | 175.7s | 3.79e+06 |
## Multi-Thread

Command: `./m-queens2 -m cpu -s 18`

|CPU|Cores (real/virtual)|N|Time to solve|Solutions per sec|Speedup from Single-Thread|
|---|---|---|---|---|---|
| AMD Ryzen 5 2400G | 18 | 25.2s | 2.65e+07 |6,97|

## OpenCL

Command: `./m-queens2 -m ocl -s 18`

|Device|N|Time to solve|Solutions per sec|
|---|---|---|---|
| AMD Radeon RX580 Series | 18 | 2.5s | 2.7e+08 |

# Future Improvements

Use the optimized placement mentioned in https://github.com/preusser/q27 to reduce symmetries.

[1]: (https://en.wikipedia.org/wiki/Eight_queens_puzzle)