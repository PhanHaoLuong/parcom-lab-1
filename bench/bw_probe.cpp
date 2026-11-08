// bench/bw_probe.cpp -- how fast can this machine actually move memory?
//
// The report uses this to place the convolution against its roofline. One
// convolution step reads 128 MB and writes 128 MB, i.e. exactly the traffic of
// a copy of the matrix, so a copy kernel is the fairest possible ceiling.
//
// The measurement enters ONE parallel region and runs all repetitions inside it,
// exactly like the convolution kernels do. Building the thread team inside the
// repetition loop instead costs a fork/join per repetition and depresses the
// measured bandwidth by 10-30 %, which would make the "ceiling" look lower than
// the kernel it is supposed to bound.
//
// Output is CSV on stdout: threads,pattern,gigabytes_per_second

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <omp.h>

int main(int argc, char **argv) {
  const int threads = argc > 1 ? std::atoi(argv[1]) : 0;
  if (threads > 0) omp_set_num_threads(threads);
  omp_set_dynamic(0);

  // Same footprint as the heat matrix: 4000 x 4000 doubles = 122 MiB.
  const long n = 4000L * 4000L;
  const int reps = 20;
  std::vector<double> a(static_cast<std::size_t>(n), 1.0);
  std::vector<double> b(static_cast<std::size_t>(n), 2.0);
  std::vector<double> c(static_cast<std::size_t>(n), 0.0);

  double *pa = a.data();
  double *pb = b.data();
  double *pc = c.data();

  // First touch / page mapping, untimed.
#pragma omp parallel for schedule(static)
  for (long i = 0; i < n; ++i) pc[i] = pa[i] + pb[i];

  double copy_s = 0.0, read_s = 0.0, triad_s = 0.0;
  double sink = 0.0;

  // copy: 1 read + 1 write per element -- the same traffic as one conv step
  double t0 = omp_get_wtime();
#pragma omp parallel
  {
    for (int r = 0; r < reps; ++r) {
#pragma omp for schedule(static)
      for (long i = 0; i < n; ++i) pc[i] = pa[i];
    }
  }
  copy_s = omp_get_wtime() - t0;

  // read only: how much can be pulled in when nothing is written back
  t0 = omp_get_wtime();
#pragma omp parallel
  {
    double local = 0.0;  // private to this thread, so no reduction is needed
    for (int r = 0; r < reps; ++r) {
#pragma omp for schedule(static)
      for (long i = 0; i < n; ++i) local += pa[i];
    }
#pragma omp atomic
    sink += local;
  }
  read_s = omp_get_wtime() - t0;

  // triad: 2 reads + 1 write
  t0 = omp_get_wtime();
#pragma omp parallel
  {
    for (int r = 0; r < reps; ++r) {
#pragma omp for schedule(static)
      for (long i = 0; i < n; ++i) pc[i] = pa[i] + pb[i] * 2.0;
    }
  }
  triad_s = omp_get_wtime() - t0;

  const int t = omp_get_max_threads();
  printf("%d,copy,%.2f\n", t, 2.0 * static_cast<double>(n) * 8.0 * reps / copy_s / 1e9);
  printf("%d,read,%.2f\n", t, 1.0 * static_cast<double>(n) * 8.0 * reps / read_s / 1e9);
  printf("%d,triad,%.2f\n", t, 3.0 * static_cast<double>(n) * 8.0 * reps / triad_s / 1e9);
  if (sink < 0.0) std::printf("%d,sink,0\n", t);  // keep the reduction alive
  return 0;
}
