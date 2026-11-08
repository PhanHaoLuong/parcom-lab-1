#include "heat_matrix.h"

#include <charconv>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <system_error>
#include <vector>

namespace heat {
namespace {

/// Parses one floating-point field starting at `p` and advances `p` past it.
/// std::from_chars has no locale, no allocation and no strtod overhead; the
/// fallback keeps the code building on toolchains without floating-point
/// from_chars (GCC < 11, older libc++).
inline double parse_double(const char *&p, const char *end) {
#if defined(__cpp_lib_to_chars) && __cpp_lib_to_chars >= 201611L
  double value = 0.0;
  const std::from_chars_result r = std::from_chars(p, end, value);
  if (r.ec != std::errc() || r.ptr == p) {
    const std::size_t n =
        static_cast<std::size_t>(end - p) < 24 ? static_cast<std::size_t>(end - p) : 24;
    throw std::runtime_error("invalid numeric value near: '" + std::string(p, n) + "'");
  }
  p = r.ptr;
  return value;
#else
  char *parse_end = nullptr;
  errno = 0;
  const double value = std::strtod(p, &parse_end);
  if (parse_end == p) {
    const std::size_t n =
        static_cast<std::size_t>(end - p) < 24 ? static_cast<std::size_t>(end - p) : 24;
    throw std::runtime_error("invalid numeric value near: '" + std::string(p, n) + "'");
  }
  p = parse_end;
  return value;
#endif
}

inline bool is_separator(char c) {
  return c == ',' || c == ' ' || c == '\t' || c == '\r';
}

}  // namespace

Image load_csv(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("could not open input file: " + path);

  file.seekg(0, std::ios::end);
  const std::streamoff length = file.tellg();
  if (length <= 0) throw std::runtime_error("input file is empty: " + path);
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(static_cast<std::size_t>(length) + 1, '\0');
  file.read(buffer.data(), length);
  if (!file) throw std::runtime_error("failed to read input file: " + path);

  const char *p = buffer.data();
  const char *const end = p + static_cast<std::size_t>(length);

  Image image;
  std::vector<double> values;
  values.reserve(static_cast<std::size_t>(4000) * 4000);

  while (p < end) {
    int columns = 0;
    while (p < end && *p != '\n') {
      if (is_separator(*p)) {
        ++p;
        continue;
      }
      values.push_back(parse_double(p, end));
      ++columns;
    }
    if (p < end) ++p;  // consume '\n'

    if (columns == 0) continue;  // blank line
    if (image.rows == 0) {
      image.cols = columns;
    } else if (columns != image.cols) {
      throw std::runtime_error(
          "row " + std::to_string(image.rows) + " of " + path + " has " +
          std::to_string(columns) + " columns, expected " + std::to_string(image.cols));
    }
    ++image.rows;
  }

  if (image.rows == 0 || image.cols == 0)
    throw std::runtime_error("CSV matrix is empty: " + path);

  const std::size_t expected =
      static_cast<std::size_t>(image.rows) * static_cast<std::size_t>(image.cols);
  if (values.size() != expected)
    throw std::runtime_error("CSV matrix is ragged: " + path);

  image.data = std::move(values);
  return image;
}

}  // namespace heat
