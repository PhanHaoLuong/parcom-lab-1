#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <omp.h>

#define ARGS double* in, double* k, double* out, int rows, int cols, int Nk, int pad, size_t total, size_t ksz
#define MAPS map(alloc: in[0:total], k[0:ksz], out[0:total])

// 0: your current kernel
static void kernel_current(ARGS) {
    #pragma omp target teams distribute parallel for collapse(2) schedule(static,1) MAPS
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) {
            double sum = 0.0;
            for (int u = 0; u < Nk; ++u)
                for (int v = 0; v < Nk; ++v) {
                    int r = i + u - pad, c = j + v - pad;
                    if (r >= 0 && r < rows && c >= 0 && c < cols)
                        sum += in[(size_t)r * cols + c] * k[u * Nk + v];
                }
            out[(size_t)i * cols + j] = sum;
        }
}

// 1: flat loop with simd
static void kernel_flat_simd(ARGS) {
    #pragma omp target teams distribute parallel for simd MAPS
    for (long idx = 0; idx < (long)total; ++idx) {
        int i = idx / cols, j = idx % cols;
        double sum = 0.0;
        for (int u = 0; u < Nk; ++u)
            for (int v = 0; v < Nk; ++v) {
                int r = i + u - pad, c = j + v - pad;
                if (r >= 0 && r < rows && c >= 0 && c < cols)
                    sum += in[(size_t)r * cols + c] * k[u * Nk + v];
            }
        out[idx] = sum;
    }
}

// 2: rows across threads, columns across simd lanes
static void kernel_row_simd(ARGS) {
    #pragma omp target teams distribute parallel for MAPS
    for (int i = 0; i < rows; ++i) {
        #pragma omp simd
        for (int j = 0; j < cols; ++j) {
            double sum = 0.0;
            for (int u = 0; u < Nk; ++u)
                for (int v = 0; v < Nk; ++v) {
                    int r = i + u - pad, c = j + v - pad;
                    if (r >= 0 && r < rows && c >= 0 && c < cols)
                        sum += in[(size_t)r * cols + c] * k[u * Nk + v];
                }
            out[(size_t)i * cols + j] = sum;
        }
    }
}

int main() {
    const int rows = 4000, cols = 4000, Nk = 3, pad = 1;
    const size_t total = (size_t)rows * cols, ksz = 9;

    std::vector<double> in_v(total), out_v(total), ref_v(total);
    std::vector<double> k_v = {0.05,0.1,0.05, 0.1,0.4,0.1, 0.05,0.1,0.05};
    for (size_t x = 0; x < total; ++x) in_v[x] = std::fmod(x * 0.37, 100.0);
    double *in = in_v.data(), *k = k_v.data(), *out = out_v.data();

    // CPU reference
    #pragma omp parallel for
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) {
            double s = 0.0;
            for (int u = 0; u < Nk; ++u)
                for (int v = 0; v < Nk; ++v) {
                    int r = i + u - pad, c = j + v - pad;
                    if (r >= 0 && r < rows && c >= 0 && c < cols)
                        s += in[(size_t)r * cols + c] * k[u * Nk + v];
                }
            ref_v[(size_t)i * cols + j] = s;
        }

    #pragma omp target enter data map(to: in[0:total], k[0:ksz]) map(alloc: out[0:total])

    const char* names[] = {"current", "flat-simd", "row-simd"};
    for (int variant = 0; variant < 3; ++variant) {
        std::cout << "== " << names[variant] << " ==\n";
        for (int run = 0; run < 4; ++run) {
            double t0 = omp_get_wtime();
            switch (variant) {
                case 0: kernel_current(in, k, out, rows, cols, Nk, pad, total, ksz); break;
                case 1: kernel_flat_simd(in, k, out, rows, cols, Nk, pad, total, ksz); break;
                case 2: kernel_row_simd(in, k, out, rows, cols, Nk, pad, total, ksz); break;
            }
            std::cout << "  run " << run << ": " << (omp_get_wtime() - t0) * 1000 << " ms\n";
        }
        #pragma omp target update from(out[0:total])
        double maxrel = 0; size_t bad = 0;
        for (size_t x = 0; x < total; ++x) {
            if (!std::isfinite(out[x])) { ++bad; continue; }
            maxrel = std::max(maxrel, std::abs(out[x] - ref_v[x]) / std::max(1.0, std::abs(ref_v[x])));
        }
        std::cout << "  max rel error=" << maxrel << " non-finite=" << bad << "\n\n";
    }

    #pragma omp target exit data map(release: in[0:total], k[0:ksz], out[0:total])
}
