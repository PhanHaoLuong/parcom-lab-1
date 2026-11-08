#!/usr/bin/env bash
#
# run_bench.sh -- the reproducible benchmark behind the report.
#
# Every number in report/report.md comes from results/timings.csv, which this
# script writes. Run it on the machine the report is about:
#
#   ./run_bench.sh                  full sweep   (~4-6 min per pass)
#   ./run_bench.sh --quick          reduced sweep (~1 min)
#   ./run_bench.sh --passes 2       repeat the sweep, keep the best per config
#   ./run_bench.sh --gpu            also exercise the OpenMP target variant
#   ./run_bench.sh --iters 200      longer diffusion chain
#
# Results: results/timings.csv, results/bandwidth.csv, results/*.log

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

BIN="build/convo"
OUT_DIR="results"
ITERS=100
REPS=7
REPS_SWEEP=5
PASSES=1
THREADS_LIST="1 2 3 4 6 8 12"
TILE_SHAPES="16x1000 32x512 64x256 128x128 256x256 500x32"
GPU=0
QUICK=0

step() { printf '\n==> %s\n' "$*"; }
die()  { printf '\nERROR: %s\n' "$*" >&2; exit 1; }
log()  { printf '%s\n' "$*" | tee -a "$LOG"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)   sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    --iters)     ITERS="${2:?}"; shift ;;
    --reps)      REPS="${2:?}"; REPS_SWEEP="$2"; shift ;;
    --passes)    PASSES="${2:?}"; shift ;;
    --threads)   THREADS_LIST="${2:?}"; shift ;;
    --out)       OUT_DIR="${2:?}"; shift ;;
    --bin)       BIN="${2:?}"; shift ;;
    --gpu)       GPU=1 ;;
    --quick)     QUICK=1 ;;
    *)           die "unknown option '$1'" ;;
  esac
  shift
done

[[ -x "$BIN" ]] || die "$BIN not found -- run ./build.sh first"

MAX_THREADS="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
if [[ $QUICK -eq 1 ]]; then
  REPS=3; REPS_SWEEP=2; PASSES=1
  THREADS_LIST="1 2 4 $MAX_THREADS"
  TILE_SHAPES="32x512 128x128 256x256"
fi

mkdir -p "$OUT_DIR"
CSV="$OUT_DIR/timings.csv"
LOG="$OUT_DIR/bench.log"
rm -f "$CSV" "$LOG"

log "ParCom Lab 1 benchmark sweep"
log "  binary      : $BIN"
log "  started     : $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
log "  host        : $(uname -srm 2>/dev/null || echo unknown) / $MAX_THREADS cores"
log "  compiler    : $(sed -n 's/^compiler: //p' build/flags.txt 2>/dev/null || echo '?')"
log "  flags       : $(sed -n 's/^flags: //p' build/flags.txt 2>/dev/null || echo '?')"
log "  iterations  : $ITERS"
log "  reps        : $REPS (minimum reported)"
log "  passes      : $PASSES"

GPU_ARGS=()
[[ $GPU -eq 1 ]] && GPU_ARGS=(--gpu)

# ---------------------------------------------------------------------------
step "Correctness pass (every variant against the sequential result)"
if ! "$BIN" --iters "$ITERS" --reps 1 --csv "$CSV" "${GPU_ARGS[@]}" \
      > "$OUT_DIR/correctness.log" 2>&1; then
  cat "$OUT_DIR/correctness.log" >&2
  die "the correctness pass failed"
fi
grep -E 'PASS|FAIL|max rel|checksum' "$OUT_DIR/correctness.log" | sed 's/^/  /' | tee -a "$LOG"

# replicate padding must produce the same field: the border ring is already 30 C
step "Padding sanity check (constant 30 vs replicate)"
"$BIN" --iters "$ITERS" --reps 1 --pad replicate --algo all \
      > "$OUT_DIR/padding_replicate.log" 2>&1
c1=$(sed -n 's/.*checksum (sum of all cells): //p' "$OUT_DIR/correctness.log")
c2=$(sed -n 's/.*checksum (sum of all cells): //p' "$OUT_DIR/padding_replicate.log")
log "  constant 30 : ${c1:-?}"
log "  replicate   : ${c2:-?}"
if [[ -n "$c1" && "$c1" == "$c2" ]]; then
  log "  -> identical"
else
  log "  -> DIFFERENT (report this)"
fi

# ---------------------------------------------------------------------------
run_pass() {
  local pass="$1"

  step "Pass $pass/$PASSES: sequential baseline"
  "$BIN" --iters "$ITERS" --reps "$REPS" --algo seq --no-verify --csv "$CSV" >/dev/null

  step "Pass $pass/$PASSES: head to head at $MAX_THREADS threads"
  "$BIN" --iters "$ITERS" --reps "$REPS" --threads "$MAX_THREADS" \
         --csv "$CSV" "${GPU_ARGS[@]}" > "$OUT_DIR/head2head.log" 2>&1
  sed -n '/^algorithm/,$p' "$OUT_DIR/head2head.log" | grep -E 'sequential|row-parallel|tiled|gpu' \
      | sed 's/^/  /' | tee -a "$LOG"

  step "Pass $pass/$PASSES: thread scaling"
  for t in $THREADS_LIST; do
    for algo in row tiled; do
      "$BIN" --iters "$ITERS" --reps "$REPS_SWEEP" --threads "$t" --algo "$algo" \
             --no-verify --csv "$CSV" >/dev/null
    done
    log "  threads=$t done"
  done

  step "Pass $pass/$PASSES: tile shapes"
  for shape in $TILE_SHAPES; do
    rows="${shape%x*}"
    cols="${shape#*x}"
    "$BIN" --iters "$ITERS" --reps "$REPS_SWEEP" --threads "$MAX_THREADS" --algo tiled \
           --tile-rows "$rows" --tile-cols "$cols" --no-verify --csv "$CSV" >/dev/null
  done
  log "  $TILE_SHAPES done"
}

for pass in $(seq 1 "$PASSES"); do
  run_pass "$pass"
done

# ---------------------------------------------------------------------------
step "Memory bandwidth ceiling (bench/bw_probe.cpp)"
BW_CSV="$OUT_DIR/bandwidth.csv"
: > "$BW_CSV"
BW_CXX="${CXX:-$(sed -n 's/^compiler: //p' build/flags.txt 2>/dev/null | awk '{print $1}')}"
BW_CXX="${BW_CXX:-g++}"
if command -v "$BW_CXX" >/dev/null 2>&1; then
  if "$BW_CXX" -O3 ${NATIVE_FLAG:--march=native} -fopenmp bench/bw_probe.cpp -o build/bw_probe \
        2>"$OUT_DIR/bw_build.log"; then
    # two rounds, keep the best throughput per thread count
    : > "$BW_CSV.raw"
    for round in 1 2; do
      for t in $THREADS_LIST; do
        ./build/bw_probe "$t" >> "$BW_CSV.raw"
      done
    done
    awk -F, '{ key = $1 "|" $2; if (!(key in best) || $3 + 0 > best[key]) best[key] = $3 + 0 }
      END { for (k in best) { split(k, p, "|"); printf "%s,%s,%.2f\n", p[1], p[2], best[k] } }' \
      "$BW_CSV.raw" | sort -t, -k1,1n -k2,2 > "$BW_CSV"
    rm -f "$BW_CSV.raw"
    awk -F, '{ printf "  threads=%-2s %-6s %6.2f GB/s\n", $1, $2, $3 }' "$BW_CSV" | tee -a "$LOG"
  else
    log "  skipped: bw_probe failed to build (see $OUT_DIR/bw_build.log)"
  fi
else
  log "  skipped: no compiler on PATH"
fi

# ---------------------------------------------------------------------------
step "Summary (best of all passes)"
awk -F, -v iters="$ITERS" '
  NR > 1 && $8 != "" {
    key = $8 "|" $7 "|" $9
    if (!(key in best) || $10 + 0 < best[key]) best[key] = $10 + 0
  }
  END {
    printf "  %-8s %8s %10s %12s %11s\n", "algorithm", "threads", "tile", "time (s)", "ms/iter"
    for (k in best) {
      split(k, p, "|")
      printf "  %-8s %8s %10s %12.4f %11.3f\n", p[1], p[2], p[3], best[k], best[k] * 1000 / iters
    }
  }' "$CSV" | (read -r header; printf '%s\n' "$header"; sort -k2,2n -k1,1 -k3,3) | tee -a "$LOG"

log ""
log "  csv : $CSV"
log "  log : $LOG"
log "  next: python3 tools/make_report.py"
