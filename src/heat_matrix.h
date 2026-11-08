// heat_matrix.h -- loader for the provided 4000 x 4000 heat-intensity CSV.
//
// The matrix is stored as one flat row-major array rather than a vector of
// vectors: the convolution kernels index it directly, the sequential and
// parallel variants then allocate identical buffers, and the 4000 tiny
// per-row allocations of the old loader disappear.

#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace heat {

struct Image {
  int rows = 0;
  int cols = 0;
  std::vector<double> data;  ///< row-major, rows * cols values

  std::size_t size() const { return data.size(); }
  double at(int i, int j) const {
    return data[static_cast<std::size_t>(i) * static_cast<std::size_t>(cols) + j];
  }
};

/// Reads a comma-separated matrix of doubles.
///
/// Accepts LF or CRLF line endings (the course dataset is CRLF), blank lines,
/// and surrounding whitespace. Throws std::runtime_error on I/O failure, on a
/// non-numeric field, or when the rows are not all the same length.
Image load_csv(const std::string &path);

}  // namespace heat
