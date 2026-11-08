#!/usr/bin/env python3
"""Build report/report.md and report/report.pdf from the benchmark output.

Pipeline:  run_bench.sh  ->  results/timings.csv, results/bandwidth.csv
           make_report.py -> report/report.md, report/figures/*.png, report/report.pdf

The prose lives in report/report_template.md; the {{PLACEHOLDERS}} in it are
replaced with tables and numbers derived from the measurements, so re-running the
benchmark on another machine regenerates a correct report.

Usage:  python3 tools/make_report.py [--open]
"""

from __future__ import annotations

import csv
import datetime as dt
import html
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "report"
FIG_DIR = REPORT_DIR / "figures"
RESULTS = ROOT / "results"
TEMPLATE = REPORT_DIR / "report_template.md"
OUT_MD = REPORT_DIR / "report.md"
OUT_PDF = REPORT_DIR / "report.pdf"

MATRIX_BYTES_PER_ITER = 4000 * 4000 * 8 * 2  # read + write, in bytes

# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def read_bench(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"error: {path} not found -- run ./run_bench.sh first")
    with path.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row.get("seconds")]


def read_bandwidth(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split(",")
            if len(parts) == 3 and parts[0].isdigit():
                rows.append({"threads": int(parts[0]), "pattern": parts[1], "gbps": float(parts[2])})
    return rows


def read_flags(path: Path) -> dict:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            out[key.strip()] = value.strip()
    return out


def best(rows: list[dict], algorithm: str, **filters) -> dict | None:
    candidates = [
        r for r in rows if r["algorithm"] == algorithm
        and all(str(r.get(k)) == str(v) for k, v in filters.items())
    ]
    return min(candidates, key=lambda r: float(r["seconds"])) if candidates else None


def unique_sorted(rows: list[dict], key: str, algorithm: str | None = None) -> list[str]:
    values = {
        r[key] for r in rows
        if r.get(key) not in (None, "", "-") and (algorithm is None or r["algorithm"] == algorithm)
    }
    return sorted(values, key=lambda v: (len(v), v))


def sorted_shapes(rows: list[dict]) -> list[str]:
    """Tile shapes ordered by height then width, so the sweep reads naturally."""
    shapes = unique_sorted(rows, "tile", "tiled")
    return sorted(shapes, key=lambda s: tuple(int(x) for x in s.split("x")))


def verification(rows: list[dict]) -> dict:
    """Best measured row per algorithm among those that were actually verified."""
    out = {}
    for algo in ("seq", "row", "tiled", "gpu"):
        candidates = [r for r in rows if r["algorithm"] == algo and float(r["max_rel_err"]) >= 0]
        if candidates:
            out[algo] = min(candidates, key=lambda r: float(r["seconds"]))
    return out


def gbps(seconds: float, iters: int) -> float:
    """Effective traffic (read+write) achieved by one variant."""
    return MATRIX_BYTES_PER_ITER * iters / seconds / 1e9


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------


def table_environment(rows: list[dict], flags: dict) -> str:
    any_row = rows[0]
    threads = max(int(r["threads"]) for r in rows if r["threads"].isdigit())
    iters = any_row["iters"]
    lines = [
        "| Property | Value |",
        "| --- | --- |",
        f"| Host | `{any_row['host']}` |",
        f"| Compiler | `{flags.get('compiler', any_row['compiler'])}` |",
        f"| Build flags | `{flags.get('flags', 'n/a')}` |",
        f"| OpenMP | `{any_row['openmp']}` (201511 = OpenMP 4.5) |",
        f"| Logical CPUs | {threads} |",
        f"| Precision | {any_row['precision']} (8 bytes per element) |",
        f"| Iterations | {iters} |",
        f"| Matrix | 4000 x 4000, 122.1 MiB per buffer; {MATRIX_BYTES_PER_ITER / 1e6:.0f} MB of traffic per iteration (read + write) |",
    ]
    return "\n".join(lines)


def default_tile(rows: list[dict]) -> str | None:
    """The block shape used for the thread sweep, i.e. the program's default.

    It is the shape measured at many thread counts; the tile-shape sweep measures
    each shape only once.
    """
    counts: dict[str, int] = {}
    for row in rows:
        if row["algorithm"] == "tiled" and row["tile"] not in ("", "-"):
            counts[row["tile"]] = counts.get(row["tile"], 0) + 1
    return max(counts, key=counts.get) if counts else None


def table_head2head(rows: list[dict], seq_s: float, checked: dict, max_threads: int,
                    tile_default: str | None) -> str:
    """Like-for-like comparison: every variant at the same thread count.

    The thread-scaling table shows where each variant peaks; this table is the
    honest head-to-head at the machine's full thread count, with the tiled variant
    at its default block shape.
    """
    order = {"seq": 0, "row": 1, "tiled": 2, "gpu": 3}
    labels = {"seq": "sequential", "row": "row-parallel", "tiled": "tiled", "gpu": "gpu (offload)"}
    picks = []
    for algo in ("seq", "row", "tiled", "gpu"):
        entry = None
        if algo == "tiled" and tile_default:
            entry = best(rows, algo, threads=str(max_threads), tile=tile_default)
        entry = entry or best(rows, algo, threads=str(max_threads)) or best(rows, algo)
        if entry:
            picks.append(entry)
    picks.sort(key=lambda r: order.get(r["algorithm"], 9))
    lines = [
        "| Variant | Threads | Tile | Time (s) | ms/iter | Speed-up | Verified |",
        "| --- | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]
    for row in picks:
        seconds = float(row["seconds"])
        speedup = seq_s / seconds if seconds else 0.0
        tile = row["tile"] if row["tile"] != "-" else "—"
        verdict = checked.get(row["algorithm"])
        if verdict is None:
            verified = "—"
        else:
            err = float(verdict["max_rel_err"])
            verified = "exact" if err == 0.0 else f"{err:.1e}"
        lines.append(
            f"| {labels[row['algorithm']]} | {row['threads']} | {tile} | {seconds:.3f} | "
            f"{seconds / int(row['iters']) * 1000:.2f} | {speedup:.2f}x | {verified} |"
        )
    return "\n".join(lines)


def table_threads(rows: list[dict], seq_s: float) -> str:
    threads = unique_sorted(rows, "threads", "row")
    lines = [
        "| Threads | row-parallel (ms/iter) | row speed-up | tiled (ms/iter) | tiled speed-up |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for thread in threads:
        row = best(rows, "row", threads=thread)
        tiled = best(rows, "tiled", threads=thread)
        if not row:
            continue
        row_ms = float(row["seconds"]) / int(row["iters"]) * 1000
        cells = [thread, f"{row_ms:.2f}", f"{seq_s / float(row['seconds']):.2f}x"]
        if tiled:
            tiled_ms = float(tiled["seconds"]) / int(tiled["iters"]) * 1000
            cells += [f"{tiled_ms:.2f}", f"{seq_s / float(tiled['seconds']):.2f}x"]
        else:
            cells += ["—", "—"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_tiles(rows: list[dict], seq_s: float) -> str:
    shapes = sorted_shapes(rows)
    lines = ["| Tile shape | Blocks | Time (s) | ms/iter | Speed-up vs sequential |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for shape in shapes:
        entry = best(rows, "tiled", tile=shape)
        if not entry:
            continue
        height, width = (int(x) for x in shape.split("x"))
        blocks = -(-4000 // height) * -(-4000 // width)
        seconds = float(entry["seconds"])
        lines.append(
            f"| {shape} | {blocks} | {seconds:.3f} | {seconds / int(entry['iters']) * 1000:.2f} | "
            f"{seq_s / seconds:.2f}x |"
        )
    return "\n".join(lines)


def table_verify(checked: dict) -> str:
    lines = ["| Variant | Max relative error vs sequential | Result |", "| --- | ---: | :--- |"]
    for algo, label in (("row", "row-parallel"), ("tiled", "tiled"), ("gpu", "gpu (offload)")):
        entry = checked.get(algo)
        if not entry:
            continue
        err = float(entry["max_rel_err"])
        if err == 0.0:
            lines.append(f"| {label} | 0 (bit-identical) | PASS |")
        else:
            lines.append(f"| {label} | {err:.2e} | {'PASS' if int(entry['bad']) == 0 else 'FAIL'} |")
    lines.append("")
    lines.append(
        "Because every variant runs the same per-pixel expression in the same order, "
        "the parallel results are not merely within tolerance, they are bit-identical; "
        "any non-zero error would therefore indicate a genuine decomposition bug."
    )
    return "\n".join(lines)


def table_bandwidth(bw: list[dict], rows: list[dict]) -> str:
    if not bw:
        return ("_Bandwidth probe was not run on this machine; run `./run_bench.sh` "
                "to generate it._")
    max_threads = max(row["threads"] for row in bw)
    one = {r["pattern"]: r["gbps"] for r in bw if r["threads"] == 1}
    many = {r["pattern"]: r["gbps"] for r in bw if r["threads"] == max_threads}
    lines = [
        f"| Pattern | 1 thread (GB/s) | {max_threads} threads (GB/s) |",
        "| --- | ---: | ---: |",
    ]
    for pattern, label in (
        ("copy", "copy (1 read + 1 write) — one convolution step"),
        ("triad", "triad (2 reads + 1 write)"),
        ("read", "read only"),
    ):
        if pattern in many:
            lines.append(f"| {label} | {one.get(pattern, 0):.2f} | {many[pattern]:.2f} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.autolayout": True,
        "legend.frameon": False,
        "svg.fonttype": "none",
    })
    return plt


def fig_speedup(plt, rows: list[dict], seq_s: float) -> str:
    threads = [int(t) for t in unique_sorted(rows, "threads", "row")]
    row_series, tiled_series = [], []
    for thread in threads:
        row = best(rows, "row", threads=str(thread))
        tiled = best(rows, "tiled", threads=str(thread))
        row_series.append(seq_s / float(row["seconds"]) if row else None)
        tiled_series.append(seq_s / float(tiled["seconds"]) if tiled else None)

    fig, ax = plt.subplots(figsize=(6.0, 3.1))
    ax.plot(threads, threads, ":", color="#9aa4b2", label="ideal (linear)")
    ax.plot(threads, row_series, "o-", color="#1f5fa9", label="row-parallel")
    ax.plot(threads, tiled_series, "s--", color="#c2410c", label="tiled 32x512")
    ax.set_xlabel("OpenMP threads")
    ax.set_ylabel("speed-up vs sequential")
    ax.set_xticks(threads)
    ax.legend(loc="upper left")
    path = FIG_DIR / "fig_speedup.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return "figures/fig_speedup.png"


def fig_tiles(plt, rows: list[dict], seq_s: float) -> str:
    shapes = sorted_shapes(rows)
    values = [(s, float(best(rows, "tiled", tile=s)["seconds"])) for s in shapes]
    values = [(s, v) for s, v in values if v]
    labels = [s for s, _ in values]
    seconds = [v for _, v in values]
    fig, ax = plt.subplots(figsize=(6.0, 2.9))
    bars = ax.bar(labels, seconds, color=["#1f5fa9" if s == "32x512" else "#93b4d8" for s in labels])
    ax.set_xlabel("tile shape (rows x columns)")
    ax.set_ylabel("time per 100 iterations (s)")
    ax.tick_params(axis="x", rotation=0)
    for bar, value in zip(bars, seconds):
        ax.annotate(f"{value:.2f}", (bar.get_x() + bar.get_width() / 2, value),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7)
    ax.set_ylim(0, max(seconds) * 1.18)
    path = FIG_DIR / "fig_tiles.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return "figures/fig_tiles.png"


def fig_bandwidth(plt, bw: list[dict], rows: list[dict], seq_s: float) -> str:
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    labels, values, colors = [], [], []
    if bw:
        max_threads = max(r["threads"] for r in bw)
        one = {r["pattern"]: r["gbps"] for r in bw if r["threads"] == 1}
        many = {r["pattern"]: r["gbps"] for r in bw if r["threads"] == max_threads}
        for key, label in (("copy", "copy"), ("triad", "triad"), ("read", "read only")):
            if key not in many:
                continue
            labels.append(f"{label}\n1 thread")
            values.append(one.get(key, 0.0))
            colors.append("#c9d6e5")
            labels.append(f"{label}\n{max_threads} threads")
            values.append(many[key])
            colors.append("#7fa4cc")
    # the convolution itself, measured with the same read + write traffic
    seq_row = best(rows, "seq")
    if seq_row:
        labels.append("convolution\nsequential")
        values.append(gbps(float(seq_row["seconds"]), int(seq_row["iters"])))
        colors.append("#f0b37e")
    row = best(rows, "row")
    if row:
        labels.append(f"convolution\nrow {row['threads']}T")
        values.append(gbps(float(row["seconds"]), int(row["iters"])))
        colors.append("#c2410c")

    bars = ax.bar(labels, values, color=colors)
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:.1f}", (bar.get_x() + bar.get_width() / 2, value),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7)
    ax.set_ylabel("GB/s of useful traffic")
    ax.set_ylim(0, max(values) * 1.18)
    ax.tick_params(axis="x", labelsize=7)
    path = FIG_DIR / "fig_bw.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return "figures/fig_bw.png"


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def find_browser() -> str | None:
    candidates = []
    if os.name == "nt":
        for base in (os.environ.get("PROGRAMFILES(X86)", ""), os.environ.get("PROGRAMFILES", ""),
                     os.environ.get("LOCALAPPDATA", "")):
            if not base:
                continue
            candidates += [
                Path(base) / "Microsoft/Edge/Application/msedge.exe",
                Path(base) / "Google/Chrome/Application/chrome.exe",
            ]
    else:
        for name in ("google-chrome", "chromium", "chromium-browser", "microsoft-edge",
                     "google-chrome-stable", "brave-browser"):
            found = shutil.which(name)
            if found:
                candidates.append(Path(found))
        candidates += [Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
                       Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge")]
    for path in candidates:
        if path and Path(path).exists():
            return str(path)
    return None


CSS = """
@page { size: A4; margin: 17mm 15mm 16mm 15mm; }
body { font-family: "Segoe UI", Inter, system-ui, -apple-system, sans-serif;
       font-size: 10pt; line-height: 1.45; color: #16202c; max-width: 100%; }
h1 { font-size: 19pt; margin: 0 0 4px 0; color: #10243d; }
h2 { font-size: 13pt; margin: 18px 0 6px 0; color: #10243d;
     border-bottom: 1px solid #d6dde7; padding-bottom: 3px; page-break-after: avoid; }
h3 { font-size: 11pt; margin: 14px 0 4px 0; color: #1f3a5f; page-break-after: avoid; }
p, li { orphans: 3; widows: 3; }
hr { border: 0; border-top: 1px solid #d6dde7; margin: 14px 0; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 12px 0;
        font-size: 8.6pt; page-break-inside: avoid; }
th, td { border: 1px solid #d6dde7; padding: 3.5px 6px; text-align: left; vertical-align: top; }
th { background: #eef2f7; font-weight: 600; }
tr:nth-child(even) td { background: #fafbfc; }
code { font-family: "Cascadia Mono", "JetBrains Mono", Consolas, monospace;
       font-size: 8.6pt; background: #f4f6f8; padding: 1px 3px; border-radius: 3px; }
pre { background: #f7f9fb; border-left: 3px solid #9db4d0; padding: 8px 10px;
      font-size: 8.6pt; overflow-x: auto; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
img { max-width: 100%; margin: 6px 0 10px 0; page-break-inside: avoid; }
blockquote { margin: 8px 0; padding: 6px 12px; border-left: 3px solid #c8b273;
             background: #fdfaf0; }
strong { color: #10243d; }
"""


def build_pdf(md_text: str) -> None:
    import markdown

    body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists"])
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>ParCom Lab 1 report</title>
<style>{CSS}</style></head><body>{body}</body></html>"""
    # Images use paths relative to report/, so the HTML has to live there too
    # for the browser to resolve them.
    html_path = REPORT_DIR / "report.html"
    with html_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(document)

    browser = find_browser()
    if not browser:
        print("warning: no Chromium/Edge found, leaving report.md only")
        return

    if OUT_PDF.exists():
        OUT_PDF.unlink()
    cmd = [
        browser,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--virtual-time-budget=10000",
        f"--print-to-pdf={OUT_PDF}",
        html_path.resolve().as_uri(),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if not OUT_PDF.exists():
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        print("warning: PDF generation failed")
    else:
        print(f"wrote {OUT_PDF.relative_to(ROOT)} ({OUT_PDF.stat().st_size / 1024:.0f} KiB)")


def main() -> int:
    rows = read_bench(RESULTS / "timings.csv")
    bw = read_bandwidth(RESULTS / "bandwidth.csv")
    flags = read_flags(ROOT / "build" / "flags.txt")

    seq = best(rows, "seq")
    if not seq:
        sys.exit("error: no sequential measurement in results/timings.csv")
    seq_s = float(seq["seconds"])
    iters = int(seq["iters"])
    checked = verification(rows)

    max_threads = max(int(r["threads"]) for r in rows if r["threads"].isdigit())

    # Head-to-head numbers: every variant at the full thread count, so the
    # comparison is like for like. The thread table shows the real optimum.
    tile_default = default_tile(rows)
    row_h2h = best(rows, "row", threads=str(max_threads)) or best(rows, "row")
    tiled_h2h = (best(rows, "tiled", threads=str(max_threads), tile=tile_default)
                 if tile_default else None) or best(rows, "tiled")
    row_best = best(rows, "row")
    best_row_threads = row_best["threads"] if row_best else "n/a"

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt = setup_matplotlib()
    fig_speedup_path = fig_speedup(plt, rows, seq_s)
    fig_tiles_path = fig_tiles(plt, rows, seq_s)
    fig_bw_path = fig_bandwidth(plt, bw, rows, seq_s)

    copy_ceiling = 0.0
    if bw:
        many = [r for r in bw if r["threads"] == max(r["threads"] for r in bw)]
        copy_ceiling = next((r["gbps"] for r in many if r["pattern"] == "copy"), 0.0)

    row_gbps = gbps(float(row_h2h["seconds"]), iters) if row_h2h else 0.0
    tiled_gbps = gbps(float(tiled_h2h["seconds"]), iters) if tiled_h2h else 0.0
    seq_gbps = gbps(seq_s, iters)

    host = rows[0]["host"]
    if "MSYS" in host.upper() or "MINGW" in host.upper() or "CYGWIN" in host.upper():
        host_note = (
            "> **Provisional measurements.** These numbers were taken on Windows "
            "(MSYS2/MinGW-w64) because no Linux machine was available at the time of "
            "writing. The course requires Linux, where the same scripts regenerate this "
            "report with native numbers: `./build.sh && ./run_bench.sh && "
            "python3 tools/make_report.py`. The analysis does not depend on the platform; "
            "the absolute timings do."
        )
    else:
        host_note = ""

    gpu_rows = [r for r in rows if r["algorithm"] == "gpu"]
    if gpu_rows:
        gpu = min(gpu_rows, key=lambda r: float(r["seconds"]))
        gpu_note = (f"Measured on this machine: {float(gpu['seconds']):.3f} s for "
                    f"{iters} iterations, {seq_s / float(gpu['seconds']):.2f}x the sequential "
                    f"reference.")
    else:
        gpu_note = (
            "> The GPU path is built and guarded but was **not executed** on the machine "
            "that produced the other numbers: its GCC has no `nvptx` offload target. "
            "`./build.sh --gpu && ./run_bench.sh --gpu` fills this section in on a machine "
            "with an offload-capable toolchain. The estimates above come from the roofline "
            "(256 MB per iteration at the card's measured memory bandwidth), not from a "
            "measurement, and are labelled as such."
        )

    replacements = {
        "{{DATE}}": dt.date.today().isoformat(),
        "{{STUDENT_ID}}": os.environ.get("STUDENT_ID", "________"),
        "{{ENV_TABLE}}": table_environment(rows, flags),
        "{{HOST_NOTE}}": host_note,
        "{{TABLE_HEAD2HEAD}}": table_head2head(rows, seq_s, checked, max_threads, tile_default),
        "{{TABLE_THREADS}}": table_threads(rows, seq_s),
        "{{TABLE_TILES}}": table_tiles(rows, seq_s),
        "{{TABLE_VERIFY}}": table_verify(checked),
        "{{TABLE_BW}}": table_bandwidth(bw, rows),
        "{{FIG_SPEEDUP}}": f"![Speed-up versus thread count]({fig_speedup_path})",
        "{{FIG_TILES}}": f"![Time per tile shape]({fig_tiles_path})",
        "{{FIG_BW}}": f"![Memory bandwidth ceiling versus achieved]({fig_bw_path})",
        "{{ROW_SPEEDUP}}": fmt(seq_s / float(row_h2h["seconds"]), 2) if row_h2h else "n/a",
        "{{TILED_SPEEDUP}}": fmt(seq_s / float(tiled_h2h["seconds"]), 2) if tiled_h2h else "n/a",
        "{{BEST_ROW_SPEEDUP}}": fmt(seq_s / float(row_best["seconds"]), 2) if row_best else "n/a",
        "{{BEST_ROW_THREADS}}": best_row_threads,
        "{{THREADS}}": str(max_threads),
        "{{SEQ_MS}}": fmt(seq_s / iters * 1000, 2),
        "{{ROW_MS}}": fmt(float(row_h2h["seconds"]) / iters * 1000, 2) if row_h2h else "n/a",
        "{{TILED_MS}}": fmt(float(tiled_h2h["seconds"]) / iters * 1000, 2) if tiled_h2h else "n/a",
        "{{ROW_MS_WITH_UNIT}}": f"{float(row_h2h['seconds']) / iters * 1000:.1f} ms" if row_h2h else "n/a",
        "{{SEQ_GBPS}}": fmt(seq_gbps, 2),
        "{{ROW_GBPS}}": fmt(row_gbps, 2),
        "{{TILED_GBPS}}": fmt(tiled_gbps, 2),
        "{{COPY_GBPS}}": fmt(copy_ceiling, 2) if copy_ceiling else "n/a",
        "{{GPU_NOTE}}": gpu_note,
    }

    text = TEMPLATE.read_text(encoding="utf-8")
    for key, value in replacements.items():
        text = text.replace(key, value)

    # Always LF, whatever platform generates the report: the PDF pipeline and the
    # repository both expect it, and Windows text mode would write CRLF.
    with OUT_MD.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    print(f"wrote {OUT_MD.relative_to(ROOT)}")
    build_pdf(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
