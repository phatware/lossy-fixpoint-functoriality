#!/usr/bin/env python3
"""Plot the BBO source Malus-law calibration.

For the lab QEKG source (Type-II BBO pumped at 404 nm producing
polarization-entangled biphotons at 808 nm), this script verifies the
Malus-law fringes in the cross-arm coincidence rates as the
arm-B right waveplate is rotated through 0..90 deg.  The fringe
visibility

    V = sqrt(c1**2 + c2**2) / c0

(extracted from a 3-parameter linear fit
``counts(theta) = c0 + c1 * cos(4*theta) + c2 * sin(4*theta)``,
where the 4-theta dependence comes from a HWP rotating polarization
by 2*theta) measures coincidence-path fringe contrast (equivalently
$(C_{max}-C_{min})/(C_{max}+C_{min})$ for the fitted curve). It is
sensitive to source purity, analyzer leakage, accidentals / coincidence
background, alignment, and downstream path asymmetries. It does not
set the measurement Lipschitz constant alpha_F, which the paper bounds
independently by data processing.

Data
----
``experiments/QEKG/TestData/BBO-QWP-ROTATION-TEST/experiment_rotation_02_*.csv``
(90 files; each file is one outer "stage" -- in the default
configuration, arm-A right waveplate fixed -- and contains 90 rows
stepping arm-B right waveplate from 0 deg to 90 deg in 1 deg steps).

Usage
-----
::

    python experiments/plot_bbo_malus.py
    python experiments/plot_bbo_malus.py \
        --data-dir experiments/QEKG/TestData/BBO-QWP-ROTATION-TEST-2 \
        --representative-index 45 \
        --output experiments/Images/BBO-Malus.png
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
DEFAULT_DATA_DIR = TESTDATA_DIR / "BBO-QWP-ROTATION-TEST-2"
DEFAULT_OUTPUT = IMAGES_DIR / "BBO-Malus.png"

# ---------------------------------------------------------------------------
# Defaults and file-format constants
# ---------------------------------------------------------------------------

ANGLE_START_DEFAULT = 0.0   # degrees
ANGLE_STEP_DEFAULT = 1.0    # degrees per step within a file

# Column indices verified from header inspection (same layout as the
# LCC-Test / LCC-QWP CSVs).
COL_INDEX = {
    "Ch1": 0, "Ch2": 1, "Ch3": 2, "Ch4": 3,
    "Ch1&3": 9, "Ch1&4": 10,
    "Ch2&3": 15, "Ch2&4": 16,
}
COINCIDENCE_PAIRS = ("Ch1&3", "Ch1&4", "Ch2&3", "Ch2&4")

# Pattern for parsing the "[A] RIGHT WP at X.XX degrees" type fragments.
_FIXED_RE = re.compile(r"\[([AB])\]\s+(LEFT|RIGHT)\s+WP\s+at\s+(\d+\.\d+)\s+degrees")


@dataclass
class MalusScan:
    """One BBO rotation scan: one stage, one inner-WP sweep."""

    path: Path
    fixed_settings: str        # printable description for the legend / title
    thetas_deg: list[float]
    counts: dict[str, list[int]]   # by pair label

    @property
    def thetas_rad(self) -> list[float]:
        return [t * math.pi / 180.0 for t in self.thetas_deg]


@dataclass
class MalusFit:
    """3-parameter linear fit counts ~ c0 + c1 cos(4 theta) + c2 sin(4 theta)."""

    c0: float
    c1: float
    c2: float

    @property
    def amplitude(self) -> float:
        return math.sqrt(self.c1 * self.c1 + self.c2 * self.c2)

    @property
    def visibility(self) -> float:
        if self.c0 <= 0:
            return 0.0
        return self.amplitude / self.c0

    @property
    def phase_deg(self) -> float:
        return 0.5 * math.degrees(math.atan2(-self.c2, self.c1))

    def evaluate(self, thetas_rad: list[float]) -> list[float]:
        return [self.c0 + self.c1 * math.cos(4 * t) + self.c2 * math.sin(4 * t)
                for t in thetas_rad]


# ---------------------------------------------------------------------------
# File loading and fitting
# ---------------------------------------------------------------------------


def load_malus_scan(path: Path, angle_start: float, angle_step: float) -> MalusScan:
    """Load a single rotation scan into a MalusScan."""
    with path.open(newline="") as handle:
        header = handle.readline()
        _ = handle.readline()
        _ = handle.readline()
        rows: list[list[str]] = []
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(line.split(","))

    fixed = ", ".join(
        f"{arm}-{side} WP {ang}°"
        for arm, side, ang in _FIXED_RE.findall(header)
    ) or "(see header)"

    thetas_deg = [angle_start + i * angle_step for i in range(len(rows))]
    counts: dict[str, list[int]] = {key: [] for key in COL_INDEX}
    for parts in rows:
        for key, idx in COL_INDEX.items():
            try:
                counts[key].append(int(parts[idx]))
            except (ValueError, IndexError):
                counts[key].append(0)

    return MalusScan(path=path, fixed_settings=fixed, thetas_deg=thetas_deg, counts=counts)


def fit_malus(thetas_rad: list[float], counts: list[int]) -> MalusFit:
    """Linear-LS fit of counts(theta) = c0 + c1 cos(4t) + c2 sin(4t).

    Solves the 3x3 normal equations explicitly so this stays
    dependency-free (no numpy / scipy).
    """
    n = len(thetas_rad)
    if n < 3:
        return MalusFit(c0=float(sum(counts)) / max(1, n), c1=0.0, c2=0.0)

    # Sums of design-matrix * design-matrix columns.
    sxx00 = float(n)
    sxx01 = sum(math.cos(4 * t) for t in thetas_rad)
    sxx02 = sum(math.sin(4 * t) for t in thetas_rad)
    sxx11 = sum(math.cos(4 * t) ** 2 for t in thetas_rad)
    sxx12 = sum(math.cos(4 * t) * math.sin(4 * t) for t in thetas_rad)
    sxx22 = sum(math.sin(4 * t) ** 2 for t in thetas_rad)
    sxy0 = float(sum(counts))
    sxy1 = sum(c * math.cos(4 * t) for c, t in zip(counts, thetas_rad))
    sxy2 = sum(c * math.sin(4 * t) for c, t in zip(counts, thetas_rad))

    # 3x3 symmetric normal-equation matrix; solve by Cramer's rule.
    a = [[sxx00, sxx01, sxx02],
         [sxx01, sxx11, sxx12],
         [sxx02, sxx12, sxx22]]
    b = [sxy0, sxy1, sxy2]

    def det3(m: list[list[float]]) -> float:
        return (
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        )

    da = det3(a)
    if abs(da) < 1e-12:
        return MalusFit(c0=sxy0 / sxx00, c1=0.0, c2=0.0)

    def replace_col(m, col, vec):
        return [[(vec[i] if c == col else m[i][c]) for c in range(3)] for i in range(3)]

    c0 = det3(replace_col(a, 0, b)) / da
    c1 = det3(replace_col(a, 1, b)) / da
    c2 = det3(replace_col(a, 2, b)) / da
    return MalusFit(c0=c0, c1=c1, c2=c2)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_bbo_malus(
    scans: list[MalusScan],
    representative_index: int,
    output: Path,
    title: str,
) -> tuple[Path, dict[str, MalusFit], list[float]]:
    if not scans:
        raise ValueError("plot_bbo_malus: no scans loaded")
    if not (0 <= representative_index < len(scans)):
        raise IndexError(
            f"representative-index {representative_index} out of range "
            f"(have {len(scans)} scans)"
        )
    rep = scans[representative_index]

    # Fit each pair on the representative scan.
    fits = {
        pair: fit_malus(rep.thetas_rad, rep.counts[pair])
        for pair in COINCIDENCE_PAIRS
    }

    # Fit visibility on every scan for the bottom-panel summary.
    visibilities: list[float] = []
    for scan in scans:
        # We summarise by the "best of the four pairs" visibility per
        # scan (the high-contrast detector pair on that stage).
        vals = [fit_malus(scan.thetas_rad, scan.counts[p]).visibility for p in COINCIDENCE_PAIRS]
        visibilities.append(max(vals))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        figsize=(9.0, 6.4),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 0.7]},
    )

    # ---- Top: data + Malus fits at representative stage ------------------
    pair_colors = {
        "Ch1&3": "#2a4d8f",
        "Ch1&4": "#b6412c",
        "Ch2&3": "#3f7f5f",
        "Ch2&4": "#7a4d8c",
    }
    theta_fine_deg = [i * 0.25 for i in range(int(rep.thetas_deg[-1] / 0.25) + 1)]
    theta_fine_rad = [t * math.pi / 180.0 for t in theta_fine_deg]

    for pair, color in pair_colors.items():
        ax_top.scatter(
            rep.thetas_deg,
            rep.counts[pair],
            color=color, s=10, alpha=0.55, edgecolors="none",
        )
        ax_top.plot(
            theta_fine_deg,
            fits[pair].evaluate(theta_fine_rad),
            color=color, linewidth=1.6,
            label=f"{pair}: V={fits[pair].visibility:.3f}",
        )

    ax_top.set_xlabel("Arm-B right WP angle (degrees)")
    ax_top.set_ylabel("Coincidence counts per step (1 s)")
    ax_top.set_title(
        f"{title}\nTop: cross-arm coincidence fringes -- representative stage "
        f"({rep.fixed_settings}; file {rep.path.name})",
        fontsize=10,
    )
    ax_top.legend(loc="upper right", fontsize=8.5, framealpha=0.92)
    ax_top.grid(True, linewidth=0.4, alpha=0.35)

    # ---- Bottom: visibility across all stages ----------------------------
    stage_indices = list(range(len(scans)))
    ax_bot.scatter(
        stage_indices, visibilities,
        color="#2a4d8f", s=14, alpha=0.75, edgecolors="none",
    )
    if visibilities:
        v_mean = sum(visibilities) / len(visibilities)
        ax_bot.axhline(
            v_mean,
            color="#222222", linestyle="--", linewidth=1.0,
            label=f"Mean visibility V = {v_mean:.3f}  (across {len(visibilities)} stages)",
        )

    ax_bot.set_xlabel("Stage index (outer waveplate setting)")
    ax_bot.set_ylabel("Fringe visibility")
    ax_bot.set_ylim(0.0, 1.05)
    ax_bot.set_title(
        "Bottom: best-pair Malus-fit visibility per stage "
        "(source-purity / apparatus-stability check)",
        fontsize=10,
    )
    ax_bot.legend(loc="lower right", fontsize=9, framealpha=0.92)
    ax_bot.grid(True, linewidth=0.4, alpha=0.35)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output, fits, visibilities


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
        help="Directory containing the experiment_rotation_02_*.csv files.",
    )
    parser.add_argument(
        "--glob",
        default="experiment_rotation_02_*.csv",
        help="Glob pattern selecting which files to load.",
    )
    parser.add_argument(
        "--representative-index",
        type=int,
        default=45,
        help="0-based index of the stage shown in the top panel (default: 45 ~= middle).",
    )
    parser.add_argument(
        "--angle-start",
        type=float,
        default=ANGLE_START_DEFAULT,
        help="First waveplate angle in each scan (default: 0 deg).",
    )
    parser.add_argument(
        "--angle-step",
        type=float,
        default=ANGLE_STEP_DEFAULT,
        help="Waveplate-angle step (default: 1 deg).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output figure path.",
    )
    parser.add_argument(
        "--title",
        default="BBO source Malus-law calibration",
        help="Figure title.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    files = sorted(args.data_dir.glob(args.glob))
    if not files:
        raise FileNotFoundError(f"No files matched {args.glob!r} in {args.data_dir}")
    print(f"Loading {len(files)} BBO rotation scans from {args.data_dir}...")
    scans = [load_malus_scan(p, args.angle_start, args.angle_step) for p in files]

    out, fits, visibilities = plot_bbo_malus(
        scans,
        representative_index=args.representative_index,
        output=args.output,
        title=args.title,
    )
    print(f"Wrote {out}")
    print(f"Representative-stage fitted visibilities:")
    for pair, fit in fits.items():
        print(f"  {pair}: V={fit.visibility:.4f}, phase={fit.phase_deg:+.2f} deg, c0={fit.c0:.0f}")
    if visibilities:
        v_sorted = sorted(visibilities)
        v_mean = sum(visibilities) / len(visibilities)
        print(
            f"Across {len(visibilities)} stages: "
            f"V_mean={v_mean:.4f}, V_median={v_sorted[len(v_sorted)//2]:.4f}, "
            f"V_min={v_sorted[0]:.4f}, V_max={v_sorted[-1]:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
