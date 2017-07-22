# m-queens
Very fast n-queens solver with OpenMP support

Based on code from https://rosettacode.org/wiki/N-queens_problem#C but much more optimized.

If you find a way to make the code even faster, please let me know.
Also feel free to open a Pull Request to add your own benchmarks.

# Benchmarks
## Single-Thread

Compiled with `gcc-6 -o main.bin main.c -O3 -march=native -mtune=native -std=c99 -DN=18`

|CPU|N|Time to solve|
|---|---|---|
|Intel(R) Core(TM) i7-4600M CPU@ 3.5 GHz (single core)|18|131.596274s|

## Multi-Thread

Compiled with `gcc-6 -o main.bin main.c -O3 -march=native -mtune=native -std=c99 -DN=18 -fopenmp`

|CPU|Cores (real/virtual)|N|Time to solve|Speedup from Single-Thread|
|---|---|---|---|---|
|Intel(R) Core(TM) i7-4600M CPU@ 3.4 GHz (all cores)|2/4|18|57.537617s|2,287|

# Future Improvements

Use the optimized placement mentioned in https://github.com/preusser/q27 to reduce symmetries.
