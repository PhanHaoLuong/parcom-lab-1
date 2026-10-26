#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <omp.h>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <algorithm>

#include "load_heat_matrix.h"

using Matrix = std::vector<std::vector<double>>;
template <typename T> using Flat = std::vector<T>;

template <typename T>
struct Weights { T w[9]; };

struct GPUTimings {
    double transfer_in  = 0.0;
    double kernel       = 0.0;   // all iterations combined
    double transfer_out = 0.0;
    double cleanup      = 0.0;
    double total        = 0.0;   // excludes one-time device setup
};

struct Summary {
    bool   ran = false, ok = true;
    double seq = 0, cpu = 0, gpu = 0;   // gpu excludes setup
};

// ---------------------------------------------------------------------
// Pixel functions, shared by CPU sequential, CPU parallel and GPU.
// Summation order is identical in both, so results match exactly on CPU.
// ---------------------------------------------------------------------
#pragma omp declare target
// Border pixel: bounds-checked, zero padding.
template <typename T>
static inline T pixel_border(const T* src, const T* w,
                             int rows, int cols, int i, int j) {
    T sum = 0;
    for (int u = 0; u < 3; ++u) {
        const int r = i + u - 1;
        if (r < 0 || r >= rows) continue;
        for (int v = 0; v < 3; ++v) {
            const int c = j + v - 1;
            if (c >= 0 && c < cols)
                sum += src[static_cast<size_t>(r) * cols + c] * w[u * 3 + v];
        }
    }
    return sum;
}

// Interior pixel: no bounds checks, fully unrolled 3x3.
template <typename T>
static inline T pixel_interior(const T* src, const T* w, int cols, size_t idx) {
    const size_t C = static_cast<size_t>(cols);
    T sum = 0;
    sum += src[idx - C - 1] * w[0];
    sum += src[idx - C    ] * w[1];
    sum += src[idx - C + 1] * w[2];
    sum += src[idx     - 1] * w[3];
    sum += src[idx        ] * w[4];
    sum += src[idx     + 1] * w[5];
    sum += src[idx + C - 1] * w[6];
    sum += src[idx + C    ] * w[7];
    sum += src[idx + C + 1] * w[8];
    return sum;
}
#pragma omp end declare target

// One output row on the CPU: border pixels checked, interior vectorized.
template <typename T>
static void cpu_row(const T* src, T* dst, Weights<T> wt,
                    int rows, int cols, int i) {
    const T* w = wt.w;
    const size_t base = static_cast<size_t>(i) * cols;

    if (i == 0 || i == rows - 1 || cols < 3) {
        for (int j = 0; j < cols; ++j)
            dst[base + j] = pixel_border(src, w, rows, cols, i, j);
        return;
    }

    dst[base] = pixel_border(src, w, rows, cols, i, 0);
    #pragma omp simd
    for (int j = 1; j < cols - 1; ++j)
        dst[base + j] = pixel_interior(src, w, cols, base + j);
    dst[base + cols - 1] = pixel_border(src, w, rows, cols, i, cols - 1);
}

// ---------------------------------------------------------------------
// 1. CPU sequential, N iterations
// ---------------------------------------------------------------------
template <typename T>
double sequential_iter(const Flat<T>& input, const Weights<T>& wt,
                       Flat<T>& result, int rows, int cols, int iters) {
    const size_t total = static_cast<size_t>(rows) * cols;
    Flat<T> a = input, b(total, T(0));
    T* src = a.data();
    T* dst = b.data();

    const double base = omp_get_wtime();
    for (int it = 0; it < iters; ++it) {
        for (int i = 0; i < rows; ++i)
            cpu_row(src, dst, wt, rows, cols, i);
        std::swap(src, dst);
    }
    const double elapsed = omp_get_wtime() - base;

    result.assign(src, src + total);
    return elapsed;
}

// ---------------------------------------------------------------------
// 2. CPU parallel, N iterations (one parallel region, barrier per step)
// ---------------------------------------------------------------------
template <typename T>
double parallel_iter(const Flat<T>& input, const Weights<T>& wt,
                     Flat<T>& result, int rows, int cols, int iters) {
    const size_t total = static_cast<size_t>(rows) * cols;
    Flat<T> a = input, b(total, T(0));
    T* A = a.data();
    T* B = b.data();

    const double base = omp_get_wtime();
    #pragma omp parallel
    {
        T* src = A;
        T* dst = B;
        for (int it = 0; it < iters; ++it) {
            #pragma omp for schedule(static)
            for (int i = 0; i < rows; ++i)
                cpu_row(src, dst, wt, rows, cols, i);
            std::swap(src, dst);
        }
    }
    const double elapsed = omp_get_wtime() - base;

    const T* fin = (iters % 2 == 0) ? A : B;
    result.assign(fin, fin + total);
    return elapsed;
}

// ---------------------------------------------------------------------
// 3. GPU, N iterations, data stays on the device
// ---------------------------------------------------------------------
template <typename T>
GPUTimings gpu_iter(const Flat<T>& input, const Weights<T>& wt,
                    Flat<T>& result, int rows, int cols, int iters) {
    const size_t total = static_cast<size_t>(rows) * cols;

    GPUTimings t;
    const double total_start = omp_get_wtime();

    Flat<T> a = input, b(total, T(0));
    T* A = a.data();
    T* B = b.data();

    double start = omp_get_wtime();
    #pragma omp target enter data map(to: A[0:total]) map(alloc: B[0:total])
    t.transfer_in = omp_get_wtime() - start;

    start = omp_get_wtime();
    T* src = A;
    T* dst = B;
    for (int it = 0; it < iters; ++it) {
        #pragma omp target teams distribute parallel for \
            firstprivate(wt) map(alloc: src[0:total], dst[0:total])
        for (int i = 0; i < rows; ++i) {
            const bool inner_row = (i > 0 && i < rows - 1);
            #pragma omp simd
            for (int j = 0; j < cols; ++j) {
                const size_t idx = static_cast<size_t>(i) * cols + j;
                dst[idx] = (inner_row && j > 0 && j < cols - 1)
                    ? pixel_interior(src, wt.w, cols, idx)
                    : pixel_border(src, wt.w, rows, cols, i, j);
            }
        }
        std::swap(src, dst);
    }
    t.kernel = omp_get_wtime() - start;

    T* fin = src;   // after the final swap, src holds the latest output
    start = omp_get_wtime();
    #pragma omp target update from(fin[0:total])
    t.transfer_out = omp_get_wtime() - start;

    start = omp_get_wtime();
    #pragma omp target exit data map(release: A[0:total], B[0:total])
    t.cleanup = omp_get_wtime() - start;

    result.assign(fin, fin + total);
    t.total = omp_get_wtime() - total_start;
    return t;
}

// ---------------------------------------------------------------------
// Verification: relative error vs the sequential result
// ---------------------------------------------------------------------
template <typename T>
static bool verify(const char* name, const Flat<T>& ref,
                   const Flat<T>& got, double tol) {
    double max_rel = 0.0;
    size_t bad = 0;
    for (size_t x = 0; x < ref.size(); ++x) {
        if (!std::isfinite(got[x])) { ++bad; continue; }
        const double r = static_cast<double>(ref[x]);
        const double rel = std::abs(static_cast<double>(got[x]) - r)
                         / std::max(1.0, std::abs(r));
        max_rel = std::max(max_rel, rel);
        if (rel > tol) ++bad;
    }
    std::cout << std::scientific << "  " << std::left << std::setw(13) << name
              << " max rel error: " << max_rel << "  bad: " << bad
              << (bad == 0 ? "  PASS\n" : "  FAIL\n") << std::fixed;
    return bad == 0;
}

// ---------------------------------------------------------------------
// Run all three approaches for one precision
// ---------------------------------------------------------------------
template <typename T>
Summary run_precision(const char* label, const Matrix& heat,
                      const Matrix& kernel2d, int iters, double tol) {
    const int rows = static_cast<int>(heat.size());
    const int cols = static_cast<int>(heat[0].size());

    Flat<T> input(static_cast<size_t>(rows) * cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            input[static_cast<size_t>(i) * cols + j] = static_cast<T>(heat[i][j]);

    Weights<T> wt;
    for (int u = 0; u < 3; ++u)
        for (int v = 0; v < 3; ++v)
            wt.w[u * 3 + v] = static_cast<T>(kernel2d[u][v]);

    std::cout << "=== " << label << " ===\n";

    Flat<T> res_seq, res_cpu, res_gpu;
    Summary s;
    s.ran = true;
    s.seq = sequential_iter(input, wt, res_seq, rows, cols, iters);
    s.cpu = parallel_iter(input, wt, res_cpu, rows, cols, iters);
    const GPUTimings g = gpu_iter(input, wt, res_gpu, rows, cols, iters);
    s.gpu = g.total;

    std::cout << "  CPU sequential:   " << s.seq << " s  ("
              << s.seq / iters * 1000 << " ms/iter)\n";
    std::cout << "  CPU parallel:     " << s.cpu << " s  ("
              << s.cpu / iters * 1000 << " ms/iter)\n";
    std::cout << "  GPU total:        " << g.total << " s  (kernel "
              << g.kernel << " s = " << g.kernel / iters * 1000 << " ms/iter; "
              << "in " << g.transfer_in << " s, out " << g.transfer_out
              << " s, cleanup " << g.cleanup << " s)\n";
    std::cout << "  Correctness vs sequential:\n";
    s.ok = verify("CPU parallel", res_seq, res_cpu, tol)
         & verify("GPU", res_seq, res_gpu, tol);
    std::cout << '\n';
    return s;
}

int main(int argc, char** argv) {
    try {
        const int iters = (argc > 1) ? std::atoi(argv[1]) : 100;
        const std::string mode = (argc > 2) ? argv[2] : "both";  // double|float|both
        if (iters < 1) throw std::runtime_error("iterations must be >= 1");
        if (mode != "double" && mode != "float" && mode != "both")
            throw std::runtime_error("second argument must be double, float or both");

        Matrix heat = readCSVToMatrix("heat_matrix.csv");
        if (heat.empty() || heat[0].empty())
            throw std::runtime_error("Input matrix is empty");
        const int cols = static_cast<int>(heat[0].size());
        for (const auto& row : heat)
            if (static_cast<int>(row.size()) != cols)
                throw std::runtime_error("Input matrix must be rectangular");

        const Matrix kernel2d = {
            {0.05, 0.1, 0.05},
            {0.1,  0.4, 0.1},
            {0.05, 0.1, 0.05}
        };
        if (kernel2d.size() != 3 || kernel2d[0].size() != 3)
            throw std::runtime_error("this version requires a 3x3 kernel");

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Input dimensions: " << heat.size() << " x " << cols << '\n';
        std::cout << "Iterations:       " << iters << '\n';
        std::cout << "CPU threads:      " << omp_get_max_threads() << '\n';

        // One-time device initialization, measured once, outside everything.
        double t0 = omp_get_wtime();
        #pragma omp target
        { }
        const double setup = omp_get_wtime() - t0;
        std::cout << "GPU setup (one-time): " << setup << " s\n\n";

        Summary sd, sf;
        if (mode == "double" || mode == "both")
            sd = run_precision<double>("double", heat, kernel2d, iters, 1e-12);
        if (mode == "float" || mode == "both")
            sf = run_precision<float>("float", heat, kernel2d, iters, 1e-4);

        std::cout << "=== Summary (" << iters << " iterations) ===\n";
        std::cout << std::left << std::setw(10) << "precision"
                  << std::right << std::setw(10) << "seq (s)"
                  << std::setw(10) << "cpu (s)"
                  << std::setw(10) << "gpu (s)"
                  << std::setw(14) << "gpu vs seq"
                  << std::setw(14) << "gpu vs cpu"
                  << std::setw(18) << "gpu+setup vs cpu" << '\n';
        auto row = [&](const char* n, const Summary& s) {
            if (!s.ran) return;
            std::cout << std::left << std::setw(10) << n << std::right
                      << std::setw(10) << s.seq << std::setw(10) << s.cpu
                      << std::setw(10) << s.gpu
                      << std::setw(13) << s.seq / s.gpu << "x"
                      << std::setw(13) << s.cpu / s.gpu << "x"
                      << std::setw(17) << s.cpu / (s.gpu + setup) << "x\n";
        };
        row("double", sd);
        row("float", sf);

        return (sd.ok && sf.ok) ? 0 : 2;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}

