#include "gpu_offload.h"

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <cstring>

namespace conv {
namespace {

#if defined(PARCOM_GPU) && defined(_OPENMP) && (_OPENMP >= 201511)
#define PARCOM_HAS_OFFLOAD 1
#else
#define PARCOM_HAS_OFFLOAD 0
#endif

}  // namespace

bool gpu_available(std::string *why_not) {
  const auto fail = [why_not](const char *msg) {
    if (why_not) *why_not = msg;
    return false;
  };

#if !PARCOM_HAS_OFFLOAD
  return fail(
      "this binary was built without OpenMP offload support "
      "(rebuild with ./build.sh --gpu)");
#else
  const int devices = omp_get_num_devices();
  if (devices < 1)
    return fail("no OpenMP target device found (driver missing, or no GPU)");

  // A device can be reported and still be unusable; probe it for real.
  int usable = 0;
#pragma omp target map(tofrom : usable)
  { usable = (omp_is_initial_device() == 0) ? 1 : 0; }
  if (!usable)
    return fail("the OpenMP target region fell back to the host");
  return true;
#endif
}

bool iterate_gpu(const std::vector<double> &input, const Weights3x3<double> &wt,
                 std::vector<double> &out, int rows, int cols, int iters,
                 double pad, PadMode mode, GpuTimings &t) {
  t = GpuTimings{};
#if !PARCOM_HAS_OFFLOAD
  (void)input;
  (void)wt;
  (void)out;
  (void)rows;
  (void)cols;
  (void)iters;
  (void)pad;
  (void)mode;
  return false;
#else
  if (!gpu_available(nullptr)) return false;

  const std::size_t total = static_cast<std::size_t>(rows) * cols;
  std::vector<double> a = input;
  std::vector<double> b(total, 0.0);
  double *A = a.data();
  double *B = b.data();

  const double overall = omp_get_wtime();

  // One-time warm-up: driver context creation, module load, first launch.
  double start = omp_get_wtime();
#pragma omp target
  {
  }
  t.setup = omp_get_wtime() - start;

  start = omp_get_wtime();
#pragma omp target enter data map(to : A[0 : total]) map(alloc : B[0 : total])
  t.transfer_in = omp_get_wtime() - start;

  // The matrix stays resident on the device for all iterations; only the two
  // pointer roles swap. `map(alloc)` tells the runtime the storage is already
  // there so it must not copy it again.
  start = omp_get_wtime();
  double *src = A;
  double *dst = B;
  for (int it = 0; it < iters; ++it) {
#pragma omp target teams distribute parallel for collapse(2) map(alloc : src[0 : total], dst[0 : total]) \
    firstprivate(wt, pad, mode)
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        const std::size_t idx = static_cast<std::size_t>(i) * cols + j;
        const bool interior = (i > 0 && i < rows - 1 && j > 0 && j < cols - 1);
        dst[idx] = interior ? pixel_interior(src, wt.w, cols, idx)
                            : pixel_edge(src, wt.w, rows, cols, i, j, pad, mode);
      }
    }
    std::swap(src, dst);
  }
  t.kernel = omp_get_wtime() - start;

  start = omp_get_wtime();
#pragma omp target update from(src[0 : total])
  t.transfer_out = omp_get_wtime() - start;

  start = omp_get_wtime();
#pragma omp target exit data map(release : A[0 : total], B[0 : total])
  t.teardown = omp_get_wtime() - start;

  out.assign(src, src + total);
  t.total = omp_get_wtime() - overall;
  t.ran = true;
  return true;
#endif
}

}  // namespace conv
