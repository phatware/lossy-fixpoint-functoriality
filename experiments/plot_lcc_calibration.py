#!/usr/bin/env python3
"""Plot the liquid-crystal-compensator (LCC) voltage calibration curve.

For the lab QRNG the polarization measurement basis on arm B is set by
a liquid-crystal retarder (Thorlabs LCC25) whose retardance varies with
applied voltage.  The PID loop on the signal arm locks the LCC voltage
to the working point at which the cross-arm coincidence pairs are
balanced 50/50 (so the binary-reduced Z_4 output is unbiased).

Data: ``experiments/QEKG/TestData/LCC-Test/lcctest_03_*.csv``
(81 files; each file is one LCC angle setting and contains 301 rows,
one per LCC voltage step from 2.000 V to 2.600 V in 0.002 V steps).

What the figure shows
---------------------
Top panel: raw cross-arm coincidence counts for the four entangled
photon-pair channels (Ch1&3, Ch1&4, Ch2&3, Ch2&4) at one
representative LCC/QWP angle.  These should trace out the Malus-law
sinusoid as the LCC retardance sweeps through pi.  The crossing of
the two same-MSB pairs gives the 50/50 working voltage.

Bottom panel: the MSB asymmetry
    A(V) = ( (Ch1&3 + Ch2&4) - (Ch1&4 + Ch2&3) ) / total
computed for every available LCC angle file and overlaid.  This is
the empirical analogue of the bias the QRNG sees: it crosses zero
exactly at the working voltage and has a steep slope there, so a
small voltage offset translates into a small monobit bias -- which
is exactly the ``eta_det'' floor that Figure 4 measures.

Usage
-----
::

    python experiments/plot_lcc_calibration.py
    python experiments/plot_lcc_calibration.py \
        --data-dir experiments/QEKG/TestData/LCC-Test \
        --representative-index 40 \
        --output experiments/Images/LCC-Voltage.png
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
TESTDATA_DIR = SCRIPT_DIR / "QEKG" / "TestData"
IMAGES_DIR = SCRIPT_DIR / "Images"
DEFAULT_DATA_DIR = TESTDATA_DIR / "LCC-Test"
DEFAULT_OUTPUT = IMAGES_DIR / "LCC-Voltage.png"

# ---------------------------------------------------------------------------
# Constants matching the data file structure
# ---------------------------------------------------------------------------

V_START_DEFAULT = 2.000   # V, first voltage step
V_STEP_DEFAULT = 0.002    # V, voltage step

# Column indices for the cross-arm coincidence pairs (verified from
# header inspection).  Ch1,Ch2 are arm-A singles; Ch3,Ch4 are arm-B
# singles; the entangled-pair coincidences are the cross-arm ones.
COL_INDEX = {
    "Ch1": 0, "Ch2": 1, "Ch3": 2, "Ch4": 3,
    "Ch1&3": 9, "Ch1&4": 10,
    "Ch2&3": 15, "Ch2&4": 16,
}

# Pattern for parsing "X.XX degrees" from the file header line.
_DEG_RE = re.compile(r"(\d+\.\d+)\s*degrees")


@dataclass
class LCCScan:
    """One LCC-voltage scan at a fixed (LCC/QWP) angle."""

    path: Path
    angle_deg: float
    voltages: list[float]
    pair_counts: dict[str, list[int]]   # e.g. {"Ch1&3": [...], ...}

    @property
    def asymmetry(self) -> list[float]:
        """MSB asymmetry A(V) = ((p13+p24) - (p14+p23)) / total."""
        n = len(self.voltages)
        out: list[float] = []
        p13 = self.pair_counts["Ch1&3"]
        p14 = self.pair_counts["Ch1&4"]
        p23 = self.pair_counts["Ch2&3"]
        p24 = self.pair_counts["Ch2&4"]
        for i in range(n):
            total = p13[i] + p14[i] + p23[i] + p24[i]
            if total <= 0:
                out.append(0.0)
            else:
                out.append(((p13[i] + p24[i]) - (p14[i] + p23[i])) / total)
        return out

    def working_voltage(self) -> float | None:
        """Linear-interpolation estimate of the V at which A(V) = 0."""
        a = self.asymmetry
        for i in range(len(a) - 1):
            if a[i] == 0:
                return self.voltages[i]
            if a[i] * a[i + 1] < 0:  # sign change
                t = a[i] / (a[i] - a[i + 1])
                return self.voltages[i] + t * (self.voltages[i + 1] - self.voltages[i])
        return None


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------


def _parse_angle(header: str) -> float:
    """Extract the LCC angle (degrees) from the file header line."""
    matches = _DEG_RE.findall(header)
    if not matches:
        return float("nan")
    # The first 'X.XX degrees' in the header is the LCC angle.
    return float(matches[0])


def load_lcc_scan(path: Path, v_start: float, v_step: float) -> LCCScan:
    """Load a single LCC voltage scan from a CSV file."""
    with path.open(newline="") as handle:
        header = handle.readline()
        _ = handle.readline()  # blank
        _ = handle.readline()  # column names
        rows: list[list[str]] = []
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(line.split(","))

    voltages = [v_start + i * v_step for i in range(len(rows))]
    pair_counts: dict[str, list[int]] = {key: [] for key in COL_INDEX}
    for parts in rows:
        for key, idx in COL_INDEX.items():
            try:
                pair_counts[key].append(int(parts[idx]))
            except (ValueError, IndexError):
                pair_counts[key].append(0)

    return LCCScan(
        path=path,
        angle_deg=_parse_angle(header),
        voltages=voltages,
        pair_counts=pair_counts,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_lcc_calibration(
    scans: list[LCCScan],
    representative_index: int,
    output: Path,
    title: str,
    pid_working_point: float | None = None,
) -> Path:
    if not scans:
        raise ValueError("plot_lcc_calibration: no scans loaded")

    if not (0 <= representative_index < len(scans)):
        raise IndexError(
            f"representative-index {representative_index} out of range "
            f"(have {len(scans)} scans)"
        )
    rep = scans[representative_index]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        figsize=(9.0, 6.4),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 0.85]},
    )

    # ---- Top: raw coincidence counts at representative angle --------------
    pair_colors = {
        "Ch1&3": "#2a4d8f",
        "Ch1&4": "#b6412c",
        "Ch2&3": "#3f7f5f",
        "Ch2&4": "#7a4d8c",
    }
    for pair, color in pair_colors.items():
        ax_top.plot(
            rep.voltages,
            rep.pair_counts[pair],
            color=color,
            linewidth=1.4,
            label=pair,
        )
    ax_top.set_ylabel("Coincidence counts per step (1 s)")
    ax_top.set_title(
        f"{title}\nTop: cross-arm coincidence counts at "
        f"LCC/QWP = {rep.angle_deg:.2f}° (file {rep.path.name})",
        fontsize=10,
    )
    ax_top.legend(loc="center right", fontsize=9, framealpha=0.92)
    ax_top.grid(True, linewidth=0.4, alpha=0.35)

    # ---- Bottom: asymmetry overlay (all angles) ---------------------------
    for scan in scans:
        ax_bot.plot(
            scan.voltages,
            scan.asymmetry,
            color="#888888",
            linewidth=0.5,
            alpha=0.22,
        )

    # Highlight the representative scan
    ax_bot.plot(
        rep.voltages,
        rep.asymmetry,
        color="#b6412c",
        linewidth=2.2,
        label=f"Representative LCC/QWP angle ({rep.angle_deg:.2f}°)",
    )

    ax_bot.axhline(
        0.0,
        color="#222222",
        linestyle="--",
        linewidth=1.0,
        label="A = 0 (perfect 50/50 balance)",
    )

    if pid_working_point is not None:
        ax_bot.axvline(
            pid_working_point,
            color="#1f4e8a",
            linestyle=":",
            linewidth=1.6,
            label=f"PID lock V = {pid_working_point:.3f} V (from entangled_tests.csv)",
        )

    ax_bot.set_xlabel("LCC voltage (V)")
    ax_bot.set_ylabel("MSB asymmetry $A(V)$")
    ax_bot.set_title(
        "Bottom: MSB asymmetry "
        "$A(V) = [(C_{13}+C_{24})-(C_{14}+C_{23})] / \\Sigma C$ "
        f"for all {len(scans)} angle scans",
        fontsize=10,
    )
    ax_bot.legend(loc="upper right", fontsize=8.5, framealpha=0.92)
    ax_bot.grid(True, linewidth=0.4, alpha=0.35)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the lcctest_03_*.csv voltage-scan files.",
    )
    parser.add_argument(
        "--glob",
        default="lcctest_03_*.csv",
        help="Glob pattern selecting which files to load (default: lcctest_03_*.csv).",
    )
    parser.add_argument(
        "--representative-index",
        type=int,
        default=40,
        help="0-based index of the scan to highlight in the top panel (default: 40 ~= middle).",
    )
    parser.add_argument(
        "--v-start",
        type=float,
        default=V_START_DEFAULT,
        help="First LCC voltage in each scan (default: 2.000 V).",
    )
    parser.add_argument(
        "--v-step",
        type=float,
        default=V_STEP_DEFAULT,
        help="LCC voltage step between rows (default: 0.002 V).",
    )
    parser.add_argument(
        "--pid-working-point",
        type=float,
        default=2.300,
        help=(
            "PID setpoint to mark on the bottom panel (default: 2.300 V; "
            "matches the LCC Volts column in entangled_tests.csv)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output figure path; extension chooses the format.",
    )
    parser.add_argument(
        "--title",
        default="LCC retarder calibration: coincidence rate and MSB asymmetry vs voltage",
        help="Figure title.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    files = sorted(args.data_dir.glob(args.glob))
    if not files:
        raise FileNotFoundError(
            f"No files matched {args.glob!r} in {args.data_dir}"
        )
    print(f"Loading {len(files)} LCC voltage-scan files from {args.data_dir}...")
    scans = [load_lcc_scan(p, args.v_start, args.v_step) for p in files]

    working_vs = [s.working_voltage() for s in scans]
    working_vs = [v for v in working_vs if v is not None and math.isfinite(v)]
    if working_vs:
        mean = sum(working_vs) / len(working_vs)
        sigma = math.sqrt(
            sum((v - mean) ** 2 for v in working_vs) / max(1, len(working_vs))
        )
        print(
            f"Estimated 50/50 working voltage across {len(working_vs)} angles: "
            f"{mean:.4f} +/- {sigma:.4f} V"
        )

    out = plot_lcc_calibration(
        scans,
        representative_index=args.representative_index,
        output=args.output,
        title=args.title,
        pid_working_point=args.pid_working_point,
    )
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
