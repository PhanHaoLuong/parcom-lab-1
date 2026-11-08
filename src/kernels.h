// kernels.h -- the convolution iteration loops.
//
// Three decompositions of the same 100-step heat diffusion:
//
//   iterate_sequential   no threads at all: the baseline everything is
//                        measured against.
//   iterate_row_parallel thread-per-row-block, `omp for schedule(static)`
//                        inside ONE parallel region reused for all iterations.
//   iterate_tiled        the output is cut into T x T tiles; the tile grid is
//                        flattened with collapse(2) and handed to the threads,
//                        the inner j loop is vectorised.
//
// All three call the same per-pixel primitives (row_cpu / pixel_interior /
// pixel_edge). The arithmetic is therefore identical everywhere: only the
// mapping of work to threads differs, which is exactly what the lab's time
// comparison is supposed to isolate.

#pragma once

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include <omp.h>

#include "pixel.h"

namespace conv {

template <typename T> struct Weights3x3 {
  T w[9];
};

/// Computes the output for columns [j0, j1) of row `i`.
///
/// This is the single workhorse behind every CPU variant. Only the pixels that
/// actually sit on the matrix border pay for a bounds check; everything else
/// runs through the branch-free, vectorised interior path. A tile that touches
/// the border therefore costs the same as an interior tile plus forty-odd
/// pixels, instead of falling back to the slow path for the whole 128x128 block.
template <typename T>
inline void compute_span(const T *__restrict src, T *__restrict dst,
                         const Weights3x3<T> &wt, int rows, int cols, int i,
                         int j0, int j1, T pad, PadMode mode) {
  const T *w = wt.w;
  const std::size_t base = static_cast<std::size_t>(i) * cols;

  if (i == 0 || i == rows - 1) {  // a border row: every pixel needs a check
    for (int j = j0; j < j1; ++j)
      dst[base + j] = pixel_edge(src, w, rows, cols, i, j, pad, mode);
    return;
  }

  // Interior columns of this span: never the first or last column of the matrix.
  const int fast0 = std::max(j0, 1);
  const int fast1 = std::min(j1, cols - 1);

  for (int j = j0; j < fast0; ++j)  // left-hand border column, at most one pixel
    dst[base + j] = pixel_edge(src, w, rows, cols, i, j, pad, mode);

  if (fast0 < fast1) {
#pragma omp simd
    for (int j = fast0; j < fast1; ++j)
      dst[base + j] = pixel_interior(src, w, cols, base + j);
  }

  for (int j = std::max(fast0, fast1); j < j1; ++j)  // right-hand border column
    dst[base + j] = pixel_edge(src, w, rows, cols, i, j, pad, mode);
}

/// One whole output row: the span version with j0 = 0, j1 = cols.
template <typename T>
inline void row_cpu(const T *__restrict src, T *__restrict dst,
                    const Weights3x3<T> &wt, int rows, int cols, int i, T pad,
                    PadMode mode) {
  compute_span(src, dst, wt, rows, cols, i, 0, cols, pad, mode);
}

/// A tile that lies strictly inside the matrix needs no bounds checks at all.
/// 900 of the 1024 tiles of a 4000x4000 image with tile=128 take this path.
template <typename T>
inline void tile_interior(const T *__restrict src, T *__restrict dst,
                          const Weights3x3<T> &wt, int cols, int i0, int i1,
                          int j0, int j1) {
  for (int i = i0; i < i1; ++i) {
    const std::size_t base = static_cast<std::size_t>(i) * cols;
#pragma omp simd
    for (int j = j0; j < j1; ++j)
      dst[base + j] = pixel_interior(src, wt.w, cols, base + j);
  }
}

// ---------------------------------------------------------------------------
// 1. Sequential, `iters` iterations.
// ---------------------------------------------------------------------------
template <typename T>
double iterate_sequential(const std::vector<T> &input, const Weights3x3<T> &wt,
                          std::vector<T> &out, int rows, int cols, int iters,
                          T pad, PadMode mode) {
  const std::size_t total = static_cast<std::size_t>(rows) * cols;
  std::vector<T> a = input;
  std::vector<T> b(total, T(0));
  T *src = a.data();
  T *dst = b.data();

  const double t0 = omp_get_wtime();
  for (int it = 0; it < iters; ++it) {
    for (int i = 0; i < rows; ++i)
      row_cpu(src, dst, wt, rows, cols, i, pad, mode);
    std::swap(src, dst);  // write into the buffer just read -> double buffer
  }
  const double seconds = omp_get_wtime() - t0;

  out.assign(src, src + total);  // after the last swap `src` holds the result
  return seconds;
}

// ---------------------------------------------------------------------------
// 2. Row-parallel: one parallel region for the whole run, a barrier per
//    iteration. No thread start-up cost inside the timed loop.
// ---------------------------------------------------------------------------
template <typename T>
double iterate_row_parallel(const std::vector<T> &input,
                            const Weights3x3<T> &wt, std::vector<T> &out,
                            int rows, int cols, int iters, T pad,
                            PadMode mode) {
  const std::size_t total = static_cast<std::size_t>(rows) * cols;
  std::vector<T> a = input;
  std::vector<T> b(total, T(0));
  T *A = a.data();
  T *B = b.data();

  const double t0 = omp_get_wtime();
#pragma omp parallel
  {
    T *src = A;
    T *dst = B;
    for (int it = 0; it < iters; ++it) {
#pragma omp for schedule(static)
      for (int i = 0; i < rows; ++i)
        row_cpu(src, dst, wt, rows, cols, i, pad, mode);
      std::swap(src, dst);  // private to the thread, but every thread does it
    }
  }
  const double seconds = omp_get_wtime() - t0;

  const T *fin = (iters % 2 == 0) ? A : B;
  out.assign(fin, fin + total);
  return seconds;
}

// ---------------------------------------------------------------------------
// 3. Tiled: the output is partitioned into `tile` x `tile` blocks. collapse(2)
//    flattens the tile grid so the scheduler sees one loop of nt_i*nt_j
//    independent work items instead of `rows` rows.
//
//    Why it helps: with row-parallel, thread k sweeps its own 333-row band,
//    which is 10.6 MB of output plus a 3-row sliding window of input. Every
//    thread streams a different part of a 128 MB matrix, so the per-core
//    private caches never get to reuse a line across threads and the
//    translation/prefetch units chase twelve independent streams. Working on
//    128x128 tiles bounds each work item to 128 KB of input and 128 KB of
//    output, which fits in L2 and keeps every stream inside a couple of pages.
// ---------------------------------------------------------------------------
template <typename T>
double iterate_tiled(const std::vector<T> &input, const Weights3x3<T> &wt,
                     std::vector<T> &out, int rows, int cols, int iters,
                     int tile_rows, int tile_cols, T pad, PadMode mode) {
  const std::size_t total = static_cast<std::size_t>(rows) * cols;
  std::vector<T> a = input;
  std::vector<T> b(total, T(0));
  T *A = a.data();
  T *B = b.data();

  if (tile_rows < 1) tile_rows = 1;
  if (tile_cols < 1) tile_cols = 1;
  const int tiles_i = (rows + tile_rows - 1) / tile_rows;
  const int tiles_j = (cols + tile_cols - 1) / tile_cols;

  const double t0 = omp_get_wtime();
#pragma omp parallel
  {
    T *src = A;
    T *dst = B;
    for (int it = 0; it < iters; ++it) {
#pragma omp for collapse(2) schedule(static)
      for (int ti = 0; ti < tiles_i; ++ti) {
        for (int tj = 0; tj < tiles_j; ++tj) {
          const int i0 = ti * tile_rows;
          const int j0 = tj * tile_cols;
          const int i1 = std::min(rows, i0 + tile_rows);
          const int j1 = std::min(cols, j0 + tile_cols);
          if (i0 > 0 && i1 < rows - 1 && j0 > 0 && j1 < cols - 1) {
            tile_interior(src, dst, wt, cols, i0, i1, j0, j1);
          } else {
            for (int i = i0; i < i1; ++i)
              compute_span(src, dst, wt, rows, cols, i, j0, j1, pad, mode);
          }
        }
      }
      std::swap(src, dst);
    }
  }
  const double seconds = omp_get_wtime() - t0;

  const T *fin = (iters % 2 == 0) ? A : B;
  out.assign(fin, fin + total);
  return seconds;
}

}  // namespace conv
