// pixel.h -- the per-pixel 3x3 convolution primitives.
//
// Every convolution variant in this project (sequential, row-parallel, tiled,
// and the optional GPU offload) is built from these two functions, so all of
// them perform the same nine multiply-adds in the same order and therefore
// produce bit-identical results. When a parallel kernel disagrees with the
// sequential one, the bug is in the decomposition, never in the arithmetic.
//
// The lab specifies a baseline temperature of 30 C outside the matrix instead
// of zero, so padding is a first-class parameter here rather than a hard-coded
// zero. `PadMode::Replicate` (copy the nearest edge value) is provided as well
// because the handout mentions it; on the provided dataset both modes agree
// exactly (the border ring is 30 C and the kernel sums to 1).

#pragma once

#include <cstddef>

namespace conv {

/// How the window is extended outside the matrix.
enum class PadMode {
  Constant,  ///< out-of-range samples take the constant value (30 C here)
  Replicate  ///< out-of-range samples copy the nearest edge value
};

/// Functions below are also compiled for OpenMP target devices when the build
/// enables offloading. The macros disappear in a plain build.
#if defined(_OPENMP)
#define PARCOM_TARGET_BEGIN _Pragma("omp declare target")
#define PARCOM_TARGET_END _Pragma("omp end declare target")
#else
#define PARCOM_TARGET_BEGIN
#define PARCOM_TARGET_END
#endif

PARCOM_TARGET_BEGIN

/// Clamp an index into [0, n).
inline int clamp_index(int v, int n) {
  return v < 0 ? 0 : (v >= n ? n - 1 : v);
}

/// Interior pixel: all eight neighbours are inside the matrix, so the 3x3
/// window is fully unrolled with no bounds checks and no branches.
template <typename T>
inline T pixel_interior(const T *__restrict src, const T *__restrict w,
                        int cols, std::size_t idx) {
  const std::size_t C = static_cast<std::size_t>(cols);
  T s = src[idx - C - 1] * w[0];
  s += src[idx - C] * w[1];
  s += src[idx - C + 1] * w[2];
  s += src[idx - 1] * w[3];
  s += src[idx] * w[4];
  s += src[idx + 1] * w[5];
  s += src[idx + C - 1] * w[6];
  s += src[idx + C] * w[7];
  s += src[idx + C + 1] * w[8];
  return s;
}

/// Any pixel on the border: bounds-checked, with the out-of-range samples
/// resolved according to `mode`.
template <typename T>
inline T pixel_edge(const T *__restrict src, const T *__restrict w, int rows,
                    int cols, int i, int j, T pad, PadMode mode) {
  T sum = T(0);
  for (int u = 0; u < 3; ++u) {
    const int r = i + u - 1;
    const bool row_ok = (r >= 0 && r < rows);
    const int rr = row_ok ? r : clamp_index(r, rows);
    for (int v = 0; v < 3; ++v) {
      const int c = j + v - 1;
      const bool col_ok = (c >= 0 && c < cols);
      T sample;
      if (row_ok && col_ok) {
        sample = src[static_cast<std::size_t>(rr) * cols + c];
      } else if (mode == PadMode::Replicate) {
        sample = src[static_cast<std::size_t>(rr) * cols + clamp_index(c, cols)];
      } else {
        sample = pad;
      }
      sum += sample * w[u * 3 + v];
    }
  }
  return sum;
}

PARCOM_TARGET_END

}  // namespace conv
