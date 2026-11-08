# ParCom Lab 1 — Parallel 2D Convolution with OpenMP

CO3068 Parallel Computing. Heat diffusion after a nuclear explosion, modelled as a
100-iteration 2D convolution of a 4000 x 4000 heat field with a 3x3 Gaussian kernel,
padded with a 30 °C baseline. Three CPU implementations plus an optional GPU
offload, a reproducible benchmark sweep, and a generated report.

**The measurements, tables and figures in `report/report.pdf` are rebuilt from
`results/timings.csv` by one command, so the report always matches the machine it
was produced on.**

## Requirements

Linux (the course requires it; the same scripts also work on macOS and MSYS2):

| Need | Version | Install |
| --- | --- | --- |
| C++17 compiler with OpenMP | GCC >= 9 (12+ recommended) or Clang >= 12 | Arch: `sudo pacman -S gcc` · Debian/Ubuntu: `sudo apt install build-essential libomp-dev` · Fedora: `sudo dnf install gcc gcc-c++` |
| Bash | 4+ | present everywhere |
| `python3` + `matplotlib` (report only) | 3.8+ | `pip install matplotlib markdown` |

No CUDA toolkit is needed for the CPU comparison. GPU offloading is optional and
uses only the compiler's own `nvptx` backend.

## Quick start

```bash
./build.sh                 # detect compiler, verify OpenMP, build build/convo
./build/convo              # 100 iterations, all three CPU variants
./run_bench.sh             # full sweep -> results/timings.csv (~4-6 min)
python3 tools/make_report.py   # report/report.md + report/report.pdf
```

Put your student ID on the report with `STUDENT_ID=2212345 python3 tools/make_report.py`.

## Commands

| Command | What it does |
| --- | --- |
| `./build.sh` | native build (`-O3 -march=native`) |
| `./build.sh --no-native` | portable binary, runs on any x86-64 |
| `./build.sh --gpu` | additionally enable OpenMP target offloading |
| `./build.sh --debug --run` | `-O0 -g`, then run |
| `./build.sh --clean` | remove `build/` |
| `make`, `make run`, `make bench` | same thing through GNU make |
| `./run_bench.sh --quick` | reduced sweep (~1 min) |
| `./run_bench.sh --gpu` | sweep including the GPU variant |
| `./package.sh <ID>` | produce `Lab_1_<ID>.zip` for submission |

`convo` options (`./build/convo --help`):

```
--input FILE          heat matrix CSV (default heat_matrix.csv)
--iters N             diffusion iterations (default 100)
--algo LIST           seq,row,tiled / all
--tile N              square tile edge (default 32x512 shape)
--tile-rows N         tile height
--tile-cols N         tile width
--threads N           OpenMP threads, 0 = runtime default
--reps N              timed repetitions, best reported
--precision double|float
--pad constant|replicate
--pad-value X         constant padding value (default 30)
--seq-time X          reuse an existing sequential baseline
--gpu                 also run the OpenMP target variant
--csv FILE            append one CSV row per measurement
```

## Layout

```
build.sh              toolchain detection, OpenMP and offload probes
run_bench.sh          the sweep that every report number comes from
package.sh            submission archive
src/pixel.h           the two per-pixel primitives, used by every kernel
src/kernels.h         sequential, row-parallel, tiled iteration loops
src/gpu_offload.*     optional OpenMP target variant + device probe
src/heat_matrix.*     flat CSV loader (std::from_chars, LF/CRLF)
src/main.cpp          CLI, timing, verification, CSV output
bench/bw_probe.cpp    memory-bandwidth ceiling used in the report
tools/make_report.py  tables, figures, PDF
report/               report_template.md (prose) -> report.md + report.pdf
results/              timings.csv, bandwidth.csv, logs (generated)
```

## How the comparison stays fair

All three CPU variants call the same `pixel_interior` / `pixel_edge` functions in
the same order, so they share one arithmetic implementation and every parallel
result is **bit-identical** to the sequential one (the report asserts error = 0, not
"within tolerance"). They differ only in how pixels are mapped to threads. Each
kernel enters one parallel region for the whole 100-iteration run, so thread
start-up is paid once rather than 100 times, and buffer allocation is outside the
timed region.

## Expected result

This kernel is memory-bound: 9 multiply-adds per pixel against 128 bytes of traffic,
about 0.56 FLOP/byte. One iteration moves 256 MB, so 100 iterations move 25.6 GB
and the achievable speed-up is set by memory bandwidth, not by core count. On a
6-core/12-thread desktop the row-parallel variant typically lands around 1.1-1.3x;
the report explains this with a measured copy-kernel roofline and shows why cache
blocking cannot help while the working set is 20x the last-level cache.

## GPU notes

`--gpu` needs a compiler built with an offload target (`gcc-*-offload-nvptx` on
Debian/Ubuntu, or Clang with `-fopenmp-targets=nvptx64-nvidia-cuda`). Arch's stock
`gcc` has no offload targets; `build.sh --gpu` detects that and tells you instead of
failing obscurely. The program also checks `omp_get_num_devices()` at run time and
reports "not available" rather than silently timing host code as if it were a GPU.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| `recipe commences before first target` | CRLF line endings in `Makefile` — the repo is pinned to LF by `.gitattributes`; run `git add --renormalize .` |
| `GCC is not configured to support 'nvptx-none'` | your GCC has no nvptx offload target; build CPU-only or use Clang |
| `error: could not open input file` | run from the repository root, or pass `--input path/to/heat_matrix.csv` |
| numbers do not match the report | the report was generated on a different machine; rerun `./run_bench.sh && python3 tools/make_report.py` |
