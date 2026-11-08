#!/usr/bin/env bash
#
# build.sh -- canonical build for ParCom Lab 1.
#
# The handout asks for a Bash script that compiles the program, so this is the
# supported entry point. It detects a compiler, verifies that OpenMP actually
# works, optionally probes OpenMP offloading, and then builds build/convo.
#
#   ./build.sh                 CPU build, native tuning
#   ./build.sh --gpu           also enable OpenMP target (GPU) offloading
#   ./build.sh --gpu --arch sm_75
#   ./build.sh --no-native     portable binary (no -march=native)
#   ./build.sh --clean         remove build/
#   ./build.sh --run           build, then run 100 iterations
#
# Exit status: 0 ok, 1 usage or toolchain problem, 2 build failure.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$ROOT_DIR/build"
TARGET="$BUILD_DIR/convo"

CXX_BIN="${CXX:-}"
GPU=0
ARCH=""
NATIVE=1
DEBUG=0
CLEAN=0
RUN_AFTER=0
EXTRA_ARGS=()

# ---------------------------------------------------------------------------
# output helpers
# ---------------------------------------------------------------------------
step() { printf '\n==> %s\n' "$*"; }
info() { printf '    %s\n' "$*"; }
warn() { printf '    warning: %s\n' "$*" >&2; }
die()  { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

usage() {
  sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
}

# ---------------------------------------------------------------------------
# arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)    usage ;;
    --gpu)        GPU=1 ;;
    -a|--arch)    ARCH="${2:-}"; shift ;;
    -n|--no-native) NATIVE=0 ;;
    -d|--debug)   DEBUG=1 ;;
    -c|--clean)   CLEAN=1 ;;
    -r|--run)     RUN_AFTER=1 ;;
    --cxx)        CXX_BIN="${2:-}"; shift ;;
    --run-args)   shift; EXTRA_ARGS=("$@"); break ;;
    *)            die "unknown option '$1' (try --help)" ;;
  esac
  shift
done

if [[ $CLEAN -eq 1 ]]; then
  step "Cleaning"
  rm -rf "$BUILD_DIR"
  info "removed $BUILD_DIR"
  [[ $RUN_AFTER -eq 0 ]] && exit 0
fi

# ---------------------------------------------------------------------------
# compiler detection
# ---------------------------------------------------------------------------
step "Compiler"
if [[ -z "$CXX_BIN" ]]; then
  for candidate in g++ clang++ c++; do
    if command -v "$candidate" >/dev/null 2>&1; then
      CXX_BIN="$candidate"
      break
    fi
  done
fi
[[ -n "$CXX_BIN" ]] || die "no C++ compiler found.
    Debian/Ubuntu : sudo apt install build-essential libomp-dev
    Arch Linux    : sudo pacman -S gcc
    Fedora        : sudo dnf install gcc gcc-c++
    macOS         : brew install libomp llvm   (then ./build.sh --cxx clang++)"
command -v "$CXX_BIN" >/dev/null 2>&1 || die "'$CXX_BIN' is not an executable on PATH"

CXX_VERSION_LINE="$("$CXX_BIN" --version | head -n 1)"
info "$CXX_VERSION_LINE"

if printf '%s' "$CXX_VERSION_LINE" | grep -qi 'clang'; then
  CLANG=1
else
  CLANG=0
  GCC_MAJOR="$("$CXX_BIN" -dumpversion | cut -d. -f1)"
  if [[ "$GCC_MAJOR" =~ ^[0-9]+$ ]] && (( GCC_MAJOR < 9 )); then
    die "$CXX_BIN is GCC $GCC_MAJOR; OpenMP $((GCC_MAJOR)) cannot compile this program.
    Use GCC >= 9 (12 or newer recommended)."
  fi
  info "GCC major version: ${GCC_MAJOR:-unknown}"
fi

# ---------------------------------------------------------------------------
# OpenMP probe: trust the compiler, not the package name
# ---------------------------------------------------------------------------
step "OpenMP support"
PROBE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/parcom-probe.XXXXXX")"
cleanup() { rm -rf "$PROBE_DIR"; }
trap cleanup EXIT

cat > "$PROBE_DIR/omp.cpp" <<'CPP'
#include <cstdio>
#include <omp.h>
int main() {
#pragma omp parallel
  { }
  std::printf("%d", omp_get_max_threads());
  return 0;
}
CPP

if ! "$CXX_BIN" -fopenmp "$PROBE_DIR/omp.cpp" -o "$PROBE_DIR/omp" >"$PROBE_DIR/omp.log" 2>&1; then
  cat "$PROBE_DIR/omp.log" >&2
  die "OpenMP is not available in '$CXX_BIN'.
    Debian/Ubuntu : sudo apt install build-essential libomp-dev
    Arch Linux    : sudo pacman -S gcc
    Fedora        : sudo dnf install gcc gcc-c++
    macOS         : brew install libomp llvm, then use clang++"
fi
info "ok: compiles, links and runs ($("$PROBE_DIR/omp" 2>/dev/null || echo '?') threads visible)"

# ---------------------------------------------------------------------------
# optional offload probe
# ---------------------------------------------------------------------------
OFFLOAD_FLAGS=()
OFFLOAD_ARCH_FLAGS=()
if [[ $GPU -eq 1 ]]; then
  step "OpenMP target offloading"

  if [[ -z "$ARCH" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
      cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d ' ' || true)"
      if [[ "$cap" =~ ^[0-9]+\.[0-9]+$ ]]; then
        ARCH="sm_${cap/./}"
        info "detected compute capability $cap -> $ARCH"
      fi
    fi
    if [[ -z "$ARCH" ]]; then
      warn "could not detect a GPU; building for the compiler's default nvptx target"
    fi
  else
    info "using requested target $ARCH"
  fi

  cat > "$PROBE_DIR/target.cpp" <<'CPP'
int main() {
  int v = 1;
#pragma omp target map(tofrom : v)
  { v += 1; }
  return v == 2 ? 0 : 1;
}
CPP

  if [[ $CLANG -eq 1 ]]; then
    OFFLOAD_FLAGS=(-fopenmp-targets=nvptx64-nvidia-cuda)
    if [[ -n "$ARCH" ]]; then
      OFFLOAD_FLAGS+=("-Xopenmp-target=nvptx64-nvidia-cuda" "-march=${ARCH}")
    fi
  else
    OFFLOAD_FLAGS=(-foffload=nvptx-none)
    if [[ -n "$ARCH" ]]; then
      OFFLOAD_ARCH_FLAGS=("-foffload-options=nvptx-none=-misa=${ARCH}")
    fi
  fi

  if ! "$CXX_BIN" -fopenmp "${OFFLOAD_FLAGS[@]}" "${OFFLOAD_ARCH_FLAGS[@]}" \
        "$PROBE_DIR/target.cpp" -o "$PROBE_DIR/target" >"$PROBE_DIR/target.log" 2>&1; then
    cat "$PROBE_DIR/target.log" >&2
    if grep -qi "not configured to support" "$PROBE_DIR/target.log"; then
      die "'$CXX_BIN' was built without an nvptx offload target.
    Arch Linux : the default 'gcc' package has no offload targets; install
                 'gcc' from a toolchain built with --enable-offload-targets,
                 or use CUDA clang (see below).
    Ubuntu     : sudo apt install gcc-13-offload-nvptx gcc-13
    clang      : ./build.sh --gpu --cxx clang++   (needs -fopenmp-targets)
    Alternative: run the CPU benchmark only (./build.sh), which is what the
                 lab's timing comparison needs."
    fi
    die "the compiler cannot build OpenMP target code; see the log above.
    Run ./build.sh without --gpu for a CPU-only build."
  fi

  # Compiling for a device is not the same as having one.
  if "$PROBE_DIR/target" >/dev/null 2>&1; then
    info "offload code builds and a device is reachable"
  else
    warn "offload code builds, but no usable device answered at run time;
      the program will fall back to CPU-only results and print why"
  fi
fi

# ---------------------------------------------------------------------------
# flags
# ---------------------------------------------------------------------------
CXXFLAGS=(-std=c++17 -Wall -Wextra -Isrc)
if [[ $DEBUG -eq 1 ]]; then
  CXXFLAGS+=(-O0 -g -fno-omit-frame-pointer)
  info "debug build (-O0 -g)"
else
  CXXFLAGS+=(-O3)
fi
if [[ $NATIVE -eq 1 ]]; then
  CXXFLAGS+=(-march=native)
else
  info "portable build (no -march=native)"
fi
CXXFLAGS+=(-fopenmp)
if [[ $GPU -eq 1 ]]; then
  CXXFLAGS+=(-DPARCOM_GPU "${OFFLOAD_FLAGS[@]}" "${OFFLOAD_ARCH_FLAGS[@]}")
fi

# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
step "Building"
mkdir -p "$BUILD_DIR"

SRC_FILES=(src/main.cpp src/heat_matrix.cpp src/gpu_offload.cpp)
for f in "${SRC_FILES[@]}"; do
  [[ -f "$ROOT_DIR/$f" ]] || die "missing source file: $f"
done

CMD=("$CXX_BIN" "${CXXFLAGS[@]}" "${SRC_FILES[@]}" -o "$TARGET")
info "${CMD[*]}"
if ! ( cd "$ROOT_DIR" && "${CMD[@]}" ); then
  die "compilation failed"
fi

{
  printf 'compiler: %s\n' "$CXX_VERSION_LINE"
  printf 'flags: %s\n' "${CXXFLAGS[*]}"
  printf 'gpu: %s\n' "$GPU"
  printf 'arch: %s\n' "${ARCH:-default}"
  printf 'host: %s\n' "$(uname -srm 2>/dev/null || echo unknown)"
} > "$BUILD_DIR/flags.txt"

info "ok -> build/convo"

# ---------------------------------------------------------------------------
# optional run
# ---------------------------------------------------------------------------
if [[ $RUN_AFTER -eq 1 ]]; then
  step "Running"
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    "$TARGET" "${EXTRA_ARGS[@]}"
  else
    "$TARGET" --iters 100
  fi
fi
