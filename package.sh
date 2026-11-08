#!/usr/bin/env bash
#
# package.sh -- build the submission archive the handout asks for.
#
#   ./package.sh 2212345          -> Lab_1_2212345.zip
#   ./package.sh 2212345 --with-data   also include the 96 MB dataset
#
# The archive contains the sources, the build and benchmark scripts, the README
# and the report. Binaries, build artefacts and the local test tooling are
# excluded; the grader builds from source with ./build.sh, which the handout
# requires anyway. The dataset is excluded by default because it is provided by
# the course, not produced by us.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

STUDENT_ID="${1:-}"
WITH_DATA=0
[[ "${2:-}" == "--with-data" ]] && WITH_DATA=1

if [[ -z "$STUDENT_ID" ]]; then
  printf 'usage: ./package.sh <STUDENT_ID> [--with-data]\n' >&2
  exit 1
fi

ZIP="Lab_1_${STUDENT_ID}.zip"
[[ -f "$ZIP" ]] && rm -f "$ZIP"

INCLUDE=(
  Makefile
  build.sh
  run_bench.sh
  package.sh
  README.md
  src
  bench
  report/report.md
  report/report.pdf
  report/figures
  tools/make_report.py
  results/timings.csv
)

for entry in "${INCLUDE[@]}"; do
  [[ -e "$entry" ]] || { printf 'warning: %s is missing, skipping\n' "$entry" >&2; }
done

mapfile -t FILES < <(for entry in "${INCLUDE[@]}"; do
  [[ -e "$entry" ]] && find "$entry" -type f
done)

if [[ $WITH_DATA -eq 1 ]]; then
  FILES+=(heat_matrix.csv)
fi

if command -v zip >/dev/null 2>&1; then
  zip -q -X "$ZIP" "${FILES[@]}"
elif command -v python3 >/dev/null 2>&1; then
  # Fallback: python3 is already required for the report generator.
  python3 - "$ZIP" "${FILES[@]}" <<'PY'
import sys, zipfile
out, files = sys.argv[1], sys.argv[2:]
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
    for name in files:
        archive.write(name, name)
print(f"wrote {out} with {len(files)} files")
PY
else
  printf 'ERROR: neither zip nor python3 is available.\n' >&2
  printf '  Arch    : sudo pacman -S zip\n  Ubuntu  : sudo apt install zip\n' >&2
  exit 1
fi

printf 'wrote %s (%s files, %s)\n' "$ZIP" "${#FILES[@]}" "$(du -h "$ZIP" 2>/dev/null | cut -f1)"
printf 'contents:\n'
if command -v unzip >/dev/null 2>&1; then
  unzip -l "$ZIP" | sed 's/^/  /'
else
  printf '  %s\n' "${FILES[@]}"
fi
