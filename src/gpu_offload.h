// gpu_offload.h -- optional OpenMP target (GPU) implementation.
//
// This lives behind its own translation unit and behind the PARCOM_GPU build
// flag on purpose. A binary compiled with plain -fopenmp and no offload support
// will happily *ignore* `#pragma omp target` and run the code on the CPU, which
// would silently report CPU timings as "GPU" timings. Both a compile-time flag
// and the runtime omp_get_num_devices() check have to agree before a GPU number
// is ever printed.

#pragma once

#include <string>
#include <vector>

#include "kernels.h"

namespace conv {

struct GpuTimings {
  bool ran = false;
  double setup = 0.0;        ///< one-time device warm-up (first kernel launch)
  double transfer_in = 0.0;  ///< host -> device, once
  double kernel = 0.0;       ///< all iterations together
  double transfer_out = 0.0; ///< device -> host, once
  double teardown = 0.0;     ///< release device allocations
  double total = 0.0;        ///< setup excluded, everything else included
};

/// True only when the binary was built with offload support and a device is
/// actually usable. `why_not` receives a human-readable reason on failure.
bool gpu_available(std::string *why_not);

/// Runs `iters` iterations on the device. Returns false (and leaves `t.ran`
/// false) when no device is usable.
bool iterate_gpu(const std::vector<double> &input, const Weights3x3<double> &wt,
                 std::vector<double> &out, int rows, int cols, int iters,
                 double pad, PadMode mode, GpuTimings &t);

}  // namespace conv
