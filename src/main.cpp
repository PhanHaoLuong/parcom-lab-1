// main.cpp -- driver for the CO3068 Lab 1 heat-diffusion convolution.
//
// Runs the same N-iteration 3x3 convolution with every implemented variant,
// times each one, and verifies every parallel result against the sequential
// result. See README.md for the full usage.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include <omp.h>

#include "gpu_offload.h"
#include "heat_matrix.h"
#include "kernels.h"

namespace {

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// The 3x3 heat-diffusion kernel from the handout. It sums to 1, so a constant
/// field is a fixed point: an all-30 C region stays at 30 C forever.
constexpr double kKernel3x3[9] = {0.05, 0.1, 0.05, 0.1, 0.4, 0.1, 0.05, 0.1, 0.05};

struct Config {
  std::string input = "heat_matrix.csv";
  int iters = 100;
  int threads = 0;  ///< 0 = let OpenMP decide
  int tile_rows = 32;  ///< 2D block decomposition: 32 rows x 512 columns
  int tile_cols = 512;
  int reps = 3;
  double pad_value = 30.0;
  conv::PadMode pad_mode = conv::PadMode::Constant;
  std::string algorithms = "seq,row,tiled";
  std::string precision = "double";
  std::string csv_path;
  double seq_seconds = -1.0;  ///< externally supplied baseline (bench sweeps)
  double tol = -1.0;          ///< <0 = derive from the precision
  bool verify = true;
  bool gpu = false;
  bool quiet = false;
};

const char *kUsage =
    "usage: convo [options]\n"
    "\n"
    "  --input FILE        heat matrix CSV            (default heat_matrix.csv)\n"
    "  --iters N           diffusion iterations       (default 100)\n"
    "  --algo LIST         comma separated, any of: seq,row,tiled,all (default all)\n"
    "  --tile N            square tile edge (overrides --tile-rows/--tile-cols)\n"
    "  --tile-rows N       tile height                (default 32)\n"
    "  --tile-cols N       tile width                 (default 512)\n"
    "  --threads N         OpenMP threads, 0 = runtime default (default 0)\n"
    "  --reps N            timed repetitions, best one reported (default 3)\n"
    "  --precision P       double or float            (default double)\n"
    "  --pad MODE          constant or replicate      (default constant)\n"
    "  --pad-value X       constant padding value     (default 30)\n"
    "  --tol X             verification tolerance     (default 1e-12 double, 1e-4 float)\n"
    "  --no-verify         skip the correctness comparison\n"
    "  --seq-time X        use X as the speed-up baseline instead of measuring it\n"
    "  --gpu               also run the OpenMP target (GPU) variant [further findings]\n"
    "  --csv FILE          append one CSV row per measurement\n"
    "  --quiet             only print the results table\n"
    "  --help              this text\n";

bool contains(const std::string &list, const std::string &item) {
  std::size_t start = 0;
  while (start <= list.size()) {
    const std::size_t comma = list.find(',', start);
    const std::string token =
        list.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
    if (token == item) return true;
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  return false;
}

int parse_int(const std::string &text, const char *flag) {
  try {
    std::size_t used = 0;
    const int value = std::stoi(text, &used);
    if (used != text.size()) throw std::invalid_argument("trailing characters");
    return value;
  } catch (const std::exception &) {
    throw std::runtime_error(std::string("invalid value for ") + flag + ": '" + text + "'");
  }
}

double parse_double(const std::string &text, const char *flag) {
  try {
    std::size_t used = 0;
    const double value = std::stod(text, &used);
    if (used != text.size()) throw std::invalid_argument("trailing characters");
    return value;
  } catch (const std::exception &) {
    throw std::runtime_error(std::string("invalid value for ") + flag + ": '" + text + "'");
  }
}

Config parse_args(int argc, char **argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&](const char *flag) -> std::string {
      if (i + 1 >= argc)
        throw std::runtime_error(std::string("missing value for ") + flag);
      return argv[++i];
    };

    if (arg == "--help" || arg == "-h") {
      std::cout << kUsage;
      std::exit(0);
    } else if (arg == "--input") {
      cfg.input = value("--input");
    } else if (arg == "--iters") {
      cfg.iters = parse_int(value("--iters"), "--iters");
    } else if (arg == "--threads") {
      cfg.threads = parse_int(value("--threads"), "--threads");
    } else if (arg == "--tile") {
      cfg.tile_rows = cfg.tile_cols = parse_int(value("--tile"), "--tile");
    } else if (arg == "--tile-rows") {
      cfg.tile_rows = parse_int(value("--tile-rows"), "--tile-rows");
    } else if (arg == "--tile-cols") {
      cfg.tile_cols = parse_int(value("--tile-cols"), "--tile-cols");
    } else if (arg == "--reps") {
      cfg.reps = parse_int(value("--reps"), "--reps");
    } else if (arg == "--algo") {
      cfg.algorithms = value("--algo");
    } else if (arg == "--precision") {
      cfg.precision = value("--precision");
    } else if (arg == "--pad") {
      const std::string mode = value("--pad");
      if (mode == "constant") {
        cfg.pad_mode = conv::PadMode::Constant;
      } else if (mode == "replicate") {
        cfg.pad_mode = conv::PadMode::Replicate;
      } else {
        throw std::runtime_error("--pad must be constant or replicate");
      }
    } else if (arg == "--pad-value") {
      cfg.pad_value = parse_double(value("--pad-value"), "--pad-value");
    } else if (arg == "--tol") {
      cfg.tol = parse_double(value("--tol"), "--tol");
    } else if (arg == "--seq-time") {
      cfg.seq_seconds = parse_double(value("--seq-time"), "--seq-time");
    } else if (arg == "--csv") {
      cfg.csv_path = value("--csv");
    } else if (arg == "--no-verify") {
      cfg.verify = false;
    } else if (arg == "--gpu") {
      cfg.gpu = true;
    } else if (arg == "--quiet") {
      cfg.quiet = true;
    } else {
      throw std::runtime_error("unknown option: " + arg);
    }
  }

  if (cfg.algorithms == "all") cfg.algorithms = "seq,row,tiled";
  if (!contains(cfg.algorithms, "seq") && !contains(cfg.algorithms, "row") &&
      !contains(cfg.algorithms, "tiled"))
    throw std::runtime_error("--algo must contain at least one of seq, row, tiled");
  if (cfg.precision != "double" && cfg.precision != "float")
    throw std::runtime_error("--precision must be double or float");
  if (cfg.iters < 1) throw std::runtime_error("--iters must be >= 1");
  if (cfg.reps < 1) throw std::runtime_error("--reps must be >= 1");
  if (cfg.tile_rows < 1) throw std::runtime_error("--tile-rows must be >= 1");
  if (cfg.tile_cols < 1) throw std::runtime_error("--tile-cols must be >= 1");
  if (cfg.threads < 0) throw std::runtime_error("--threads must be >= 0");
  return cfg;
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

std::string compiler_name() {
#if defined(__clang__)
  return std::string("clang ") + __clang_version__;
#elif defined(__GNUC__)
  return std::string("gcc ") + __VERSION__;
#else
  return "unknown compiler";
#endif
}

std::string host_name() {
  for (const char *name : {"HOSTNAME", "COMPUTERNAME"}) {
    if (const char *value = std::getenv(name)) return value;
  }
  return "-";
}

std::string utc_timestamp() {
  const std::time_t now = std::time(nullptr);
  std::tm tm_buf{};
#if defined(_WIN32)
  gmtime_s(&tm_buf, &now);
#else
  gmtime_r(&now, &tm_buf);
#endif
  char buffer[32];
  std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &tm_buf);
  return buffer;
}

std::string human_mib(std::size_t bytes) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(1)
      << static_cast<double>(bytes) / (1024.0 * 1024.0) << " MiB";
  return out.str();
}

template <typename T>
double checksum(const std::vector<T> &values) {
  long double total = 0.0L;
  for (const T v : values) total += static_cast<long double>(v);
  return static_cast<double>(total);
}

/// Compares `got` against the sequential `ref`. Returns the largest relative
/// error and the number of samples outside `tol` (or non-finite).
template <typename T>
void compare(const std::vector<T> &ref, const std::vector<T> &got, double tol,
             double &max_rel, std::size_t &bad) {
  max_rel = 0.0;
  bad = 0;
  const std::size_t n = std::min(ref.size(), got.size());
  for (std::size_t i = 0; i < n; ++i) {
    const double expected = static_cast<double>(ref[i]);
    const double actual = static_cast<double>(got[i]);
    if (!std::isfinite(actual)) {
      ++bad;
      continue;
    }
    const double rel =
        std::abs(actual - expected) / std::max(1.0, std::abs(expected));
    max_rel = std::max(max_rel, rel);
    if (rel > tol) ++bad;
  }
  if (ref.size() != got.size()) bad += (ref.size() > got.size() ? ref.size() : got.size());
}

struct Trial {
  std::string label;
  std::string algo;
  int threads = 1;
  int tile_rows = 0;
  int tile_cols = 0;
  double seconds = 0.0;
  bool checked = false;
  double max_rel = 0.0;
  std::size_t bad = 0;
  double checksum = 0.0;
};

// ---------------------------------------------------------------------------
// CSV output
// ---------------------------------------------------------------------------

class CsvWriter {
 public:
  explicit CsvWriter(const std::string &path) : path_(path) {
    if (path_.empty()) return;
    const std::filesystem::path p(path_);
    if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
    const bool exists = std::filesystem::exists(p);
    out_.open(path_, std::ios::app);
    if (!out_) throw std::runtime_error("cannot open CSV output: " + path_);
    if (!exists)
      out_ << "utc,host,compiler,openmp,precision,iters,threads,algorithm,tile,"
              "seconds,ms_per_iter,speedup,max_rel_err,bad,checksum\n";
  }

  void row(const std::string &precision, int iters, const Trial &t, double speedup) {
    if (!out_) return;
    out_ << utc_timestamp() << ',' << host_name() << ",\"" << compiler_name() << "\","
         << _OPENMP << ',' << precision << ',' << iters << ',' << t.threads << ','
         << t.algo << ',' << (t.tile_rows > 0 ? std::to_string(t.tile_rows) + "x" + std::to_string(t.tile_cols) : std::string("-"))
         << ',' << std::setprecision(6) << std::fixed << t.seconds << ','
         << std::setprecision(4) << std::fixed << t.seconds / iters * 1000.0 << ','
         << std::setprecision(4) << (speedup > 0.0 ? speedup : 0.0) << ','
         << std::setprecision(6) << std::scientific << (t.checked ? t.max_rel : -1.0)
         << ',' << std::defaultfloat << (t.checked ? t.bad : std::size_t(0)) << ','
         << std::setprecision(17) << t.checksum << '\n';
  }

 private:
  std::string path_;
  std::ofstream out_;
};

// ---------------------------------------------------------------------------
// One precision, all selected variants
// ---------------------------------------------------------------------------

template <typename T>
int run_precision(const Config &cfg, const heat::Image &image, CsvWriter &csv) {
  const int rows = image.rows;
  const int cols = image.cols;
  const std::size_t total = static_cast<std::size_t>(rows) * cols;

  std::vector<T> input(total);
  for (std::size_t i = 0; i < total; ++i) input[i] = static_cast<T>(image.data[i]);

  conv::Weights3x3<T> wt;
  double kernel_sum = 0.0;
  for (int i = 0; i < 9; ++i) {
    wt.w[i] = static_cast<T>(kKernel3x3[i]);
    kernel_sum += kKernel3x3[i];
  }

  const T pad = static_cast<T>(cfg.pad_value);
  const conv::PadMode mode = cfg.pad_mode;
  const double tol = cfg.tol > 0.0 ? cfg.tol : (std::is_same<T, double>::value ? 1e-12 : 1e-4);

  if (!cfg.quiet) {
    std::cout << "---------------------------------------------------------------"
                 "----------------\n";
    std::cout << " Precision      : " << (std::is_same<T, double>::value ? "double" : "float")
              << " (" << sizeof(T) << " bytes, " << total * sizeof(T) << " bytes per buffer)\n";
    std::cout << " Padding        : "
              << (mode == conv::PadMode::Constant ? "constant " : "replicate ")
              << (mode == conv::PadMode::Constant ? std::to_string(cfg.pad_value) : "values")
              << " outside the matrix\n";
    std::cout << " Kernel         : 3x3 gaussian, sum = " << std::fixed
              << std::setprecision(6) << kernel_sum << "\n";
    std::cout << " Iterations     : " << cfg.iters << "\n";
    std::cout << " Threads        : " << omp_get_max_threads() << " (requested: "
              << (cfg.threads > 0 ? std::to_string(cfg.threads) : std::string("runtime default"))
              << ")\n";
    std::cout << " Repetitions    : " << cfg.reps << " (best reported)\n";
    std::cout << " Tolerance      : " << std::scientific << tol << std::fixed << "\n";
  }

  std::vector<Trial> trials;

  auto timed = [&](const std::function<double(std::vector<T> &)> &run,
                   std::vector<T> &out) {
    double best = std::numeric_limits<double>::infinity();
    for (int r = 0; r < cfg.reps; ++r) best = std::min(best, run(out));
    return best;
  };

  const auto seq_run = [&](std::vector<T> &out) {
    return conv::iterate_sequential(input, wt, out, rows, cols, cfg.iters, pad, mode);
  };
  const auto row_run = [&](std::vector<T> &out) {
    return conv::iterate_row_parallel(input, wt, out, rows, cols, cfg.iters, pad, mode);
  };
  const auto tiled_run = [&](std::vector<T> &out) {
    return conv::iterate_tiled(input, wt, out, rows, cols, cfg.iters, cfg.tile_rows,
                                cfg.tile_cols, pad, mode);
  };

  const bool want_seq = contains(cfg.algorithms, "seq");
  const bool want_row = contains(cfg.algorithms, "row");
  const bool want_tiled = contains(cfg.algorithms, "tiled");

  // Reference for verification, and the speed-up baseline. When the caller did
  // not ask for the sequential run (bench sweeps), it is still computed once so
  // that the parallel variants can be checked.
  std::vector<T> reference;
  double seq_seconds = cfg.seq_seconds;
  bool seq_reported = false;

  if (want_seq) {
    Trial t;
    t.label = "sequential";
    t.algo = "seq";
    t.threads = 1;
    seq_seconds = timed(seq_run, reference);
    t.seconds = seq_seconds;
    t.checksum = checksum(reference);
    trials.push_back(t);
    seq_reported = true;
  } else if (cfg.verify || cfg.seq_seconds <= 0.0) {
    seq_seconds = seq_run(reference);
  }

  if (want_row) {
    Trial t;
    t.label = "row-parallel";
    t.algo = "row";
    t.threads = omp_get_max_threads();
    std::vector<T> out;
    t.seconds = timed(row_run, out);
    t.checksum = checksum(out);
    if (cfg.verify && !reference.empty()) {
      t.checked = true;
      compare(reference, out, tol, t.max_rel, t.bad);
    }
    trials.push_back(t);
  }

  if (want_tiled) {
    Trial t;
    t.label = "tiled " + std::to_string(cfg.tile_rows) + "x" + std::to_string(cfg.tile_cols);
    t.algo = "tiled";
    t.threads = omp_get_max_threads();
    t.tile_rows = cfg.tile_rows;
    t.tile_cols = cfg.tile_cols;
    std::vector<T> out;
    t.seconds = timed(tiled_run, out);
    t.checksum = checksum(out);
    if (cfg.verify && !reference.empty()) {
      t.checked = true;
      compare(reference, out, tol, t.max_rel, t.bad);
    }
    trials.push_back(t);
  }

  // ---- GPU variant (further findings) ------------------------------------
  // `if constexpr` keeps the double-only offload API out of the float
  // instantiation entirely.
  if (cfg.gpu) {
    if constexpr (!std::is_same<T, double>::value) {
      if (!cfg.quiet) std::cout << "\n [gpu] skipped: the offload variant is double only\n";
    } else {
      std::string why;
      if (!conv::gpu_available(&why)) {
        if (!cfg.quiet) std::cout << "\n [gpu] not available: " << why << "\n";
      } else {
        std::vector<double> out;
        conv::GpuTimings gt;
        conv::Weights3x3<double> dwt;
        for (int i = 0; i < 9; ++i) dwt.w[i] = kKernel3x3[i];
        const std::vector<double> input_d(input.begin(), input.end());
        const bool ran = conv::iterate_gpu(input_d, dwt, out, rows, cols, cfg.iters,
                                           cfg.pad_value, mode, gt);
        if (ran) {
          Trial t;
          t.label = "gpu (offload)";
          t.algo = "gpu";
          t.threads = 0;
          t.seconds = gt.total;
          t.checksum = checksum(out);
          if (cfg.verify && !reference.empty()) {
            t.checked = true;
            compare(reference, out, tol, t.max_rel, t.bad);
          }
          trials.push_back(t);
          if (!cfg.quiet) {
            std::cout << "\n [gpu] setup " << std::fixed << std::setprecision(6) << gt.setup
                      << " s (excluded), in " << gt.transfer_in << " s, kernel "
                      << gt.kernel << " s, out " << gt.transfer_out << " s, teardown "
                      << gt.teardown << " s\n";
          }
        }
      }
    }
  }

  // ---- table --------------------------------------------------------------
  std::cout << "\n";
  std::cout << std::left << std::setw(20) << "algorithm" << std::right << std::setw(8)
            << "threads" << std::setw(7) << "tile" << std::setw(13) << "time (s)"
            << std::setw(12) << "ms/iter" << std::setw(10) << "speedup" << std::setw(14)
            << "max rel err" << std::setw(8) << "status" << "\n";
  std::cout << std::string(92, '-') << "\n";

  double fastest = std::numeric_limits<double>::infinity();
  for (const Trial &t : trials) fastest = std::min(fastest, t.seconds);
  for (const Trial &t : trials) {
    const double speedup = seq_seconds > 0.0 ? seq_seconds / t.seconds : 0.0;
    std::ostringstream tile_text;
    if (t.tile_rows > 0) {
      tile_text << t.tile_rows << 'x' << t.tile_cols;
    } else {
      tile_text << "-";
    }
    std::ostringstream threads_text;
    if (t.threads > 0) {
      threads_text << t.threads;
    } else {
      threads_text << "-";
    }

    std::cout << std::left << std::setw(20) << t.label << std::right << std::setw(8)
              << threads_text.str() << std::setw(7) << tile_text.str() << std::setw(13)
              << std::fixed << std::setprecision(6) << t.seconds << std::setw(12)
              << std::fixed << std::setprecision(3) << t.seconds / cfg.iters * 1000.0
              << std::setw(9) << std::fixed << std::setprecision(2) << speedup << "x";

    if (t.checked) {
      std::cout << std::setw(14) << std::scientific << std::setprecision(2) << t.max_rel
                << std::fixed;
      std::cout << std::setw(9) << (t.bad == 0 ? "PASS" : "FAIL");
    } else {
      std::cout << std::setw(14) << "-" << std::setw(9) << "unchecked";
    }
    if (t.seconds == fastest && trials.size() > 1) std::cout << "  <- fastest";
    std::cout << "\n";

    csv.row(std::is_same<T, double>::value ? "double" : "float", cfg.iters, t, speedup);
  }

  if (!trials.empty()) {
    std::cout << std::string(92, '-') << "\n";
    std::cout << " output checksum (sum of all cells): " << std::setprecision(17) << std::fixed
              << trials.front().checksum << "\n";
  }
  if (cfg.verify && !seq_reported && !trials.empty())
    std::cout << " note: the sequential reference was measured once for verification and is "
                 "not part of this timing run\n";

  int rc = 0;
  for (const Trial &t : trials)
    if (t.checked && t.bad != 0) rc = 2;
  return rc;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const Config cfg = parse_args(argc, argv);

    if (cfg.threads > 0) omp_set_num_threads(cfg.threads);
    omp_set_dynamic(0);  // reproducible thread counts between runs

    const double load_start = omp_get_wtime();
    const heat::Image image = heat::load_csv(cfg.input);
    const double load_seconds = omp_get_wtime() - load_start;

    if (!cfg.quiet) {
      std::cout << "==============================================================="
                   "================\n";
      std::cout << " ParCom Lab 1 -- 2D convolution (heat diffusion), CO3068\n";
      std::cout << "==============================================================="
                   "================\n";
      std::cout << " Input          : " << cfg.input << " (" << image.rows << " x "
                << image.cols << ", " << human_mib(image.size() * sizeof(double))
                << " as doubles, loaded in " << std::fixed << std::setprecision(3)
                << load_seconds << " s)\n";
      std::cout << " Compiler       : " << compiler_name() << ", OpenMP " << _OPENMP
                << "\n";
      std::cout << " Build          : "
                << (conv::gpu_available(nullptr) ? "with OpenMP offload"
                                                : "CPU only (no target device)")
                << "\n";
      std::cout << " Host           : " << host_name() << ", "
                << std::thread::hardware_concurrency() << " logical CPUs\n";
    }

    CsvWriter csv(cfg.csv_path);

    int rc = 0;
    if (cfg.precision == "double")
      rc = run_precision<double>(cfg, image, csv);
    else
      rc = run_precision<float>(cfg, image, csv);
    return rc;
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n\n";
    std::cerr << kUsage;
    return 1;
  }
}
