#!/usr/bin/env python3
"""Generate a QRNG calibration figure from archived entropy-test CSVs.

This script compares the theoretical QRNG tolerance bound from the
bundled paper PDF (``LossyFixpointFunctoriality.pdf``) against
empirical bias estimates extracted from the archived per-batch Shannon
entropy CSV exports (by default ``experiments/QEKG/TestData/StatTests/*.csv``).

Mathematical model (paper, Section "QRNG as Q -> C -> C" and App. A)
-------------------------------------------------------------------
For single-pass QRNG with sub-microsecond gate window the paper gives

    epsilon_QRNG(N) <= eta_floor + c / sqrt(N),                (model)

where

  * N         is the total number of accumulated symbols (= bits, after
              the binary reduction of the Z_4 POVM used in the lab),
  * eta_floor is the hardware/detector floor.  The paper reports
              eta_floor ~ 7e-3 (median run) and ~ 2e-3 (best run).
  * c         = sqrt(2 * ln 4) / 2 ~ 0.83.  This is the finite-statistics
              coefficient that captures the typical Pinsker contribution
              from N samples of a binary-reduced Z_4 output.

The empirical bias for a single recorded H1 value is extracted via the
Pinsker-type bound (paper App. A, eq. (eq:H1bias-app)):

    epsilon_emp <= sqrt((1 - H1) * ln 2 / 2).                  (Pinsker)

How the data is laid out
------------------------
Each row in the archived CSVs (e.g. ``entangled_tests.csv``) summarizes
one cleaned 20,000-bit batch produced by the lab QRNG.  The 20,000-bit
batch size is inferred from the original Mathematica notebooks
``StatTestsENT.nb`` / ``StatTestsAMP.nb`` (search for ``len * 20000``).

Relevant columns:

  * ``H1``    Shannon entropy of the 20,000-bit batch (after MSB
              reduction of the Z_4 output).  Already in bits per symbol
              (a value of 1.0 means the batch is indistinguishable from
              an unbiased Bernoulli(1/2) source at this batch size).
  * ``Error`` Run-error code ("00" for clean batches; anything else is
              filtered out so it does not pollute the empirical curve).

How the empirical comparison curve is built
-------------------------------------------
For a target sample count N we want the empirical bias bound that
*would have been obtained from a randomly selected dataset of N bits*.
With B = 20,000 bits per batch and k = round(N / B) batches per group:

  1. Sort the clean batches as they appear in the CSV (no reordering).
  2. Split them into non-overlapping groups of k batches (last partial
     group is dropped).
  3. For each group, average the per-batch (1 - H1).  Because the
     batches are equal-size and approximately i.i.d., the average of
     per-batch (1 - H1) approximates 1 - H1 of the concatenated bits
     up to O(epsilon^4).
  4. Apply the Pinsker formula to obtain a per-group epsilon.
  5. Plot the *mean* and *spread* of those per-group epsilons against
     N = k * B.

This is the right empirical analogue of the model bound: the model says
"with N samples your bias-bound is eta + c/sqrt(N)"; the empirical curve
says "with N samples you actually observe these epsilons in the lab."

In addition, the script also plots the legacy "running prefix" curve
(cumulative average of (1 - H1) over the first k batches of the file)
because it is what previous versions of this script produced and what
appears in figures of the paper draft.  The two curves agree at the
right end of the x-axis (large N) and diverge at the left end where the
prefix curve depends on which batches happen to be first in the file.

Usage
-----
::

    # Default behaviour: reads the two entangled-test CSVs and writes
    # PNG + PDF into experiments/Images/.
    python experiments/plot_qrng_calibration.py

    # Explicit data file(s) and a single output image.
    python experiments/plot_qrng_calibration.py \
        --data experiments/experiment_data/entangled_tests.csv \
        --data experiments/experiment_data/entangled_tests2.csv \
        --output experiments/Images/qrng_calibration.png

    # See ``--help`` for all options (curve constants, batch size,
    # legacy mode, title, etc.).
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
STATTEST_DIR = SCRIPT_DIR / "QEKG" / "TestData" / "StatTests"
IMAGES_DIR = SCRIPT_DIR / "Images"

# ---------------------------------------------------------------------------
# Physical / mathematical constants used in the paper.
# ---------------------------------------------------------------------------

LN2 = math.log(2.0)

# Each archived CSV row summarizes one 20,000-bit batch (see module
# docstring and the Mathematica notebooks for the "len * 20000" idiom).
DEFAULT_BATCH_BITS = 20_000

# Default datasets: the two entangled-source archives referenced in
# the paper.  Resolved against --data-dir if --data is not used.
DEFAULT_DATASETS = ("test_stat1+2.csv", "test_stat1+2-2.csv", "test_stat1+2-3.csv")
DEFAULT_DATA_DIR = STATTEST_DIR

# Friendly names for the legend.
DISPLAY_NAMES = {
    "entangled_tests": "ENT archive I",
    "entangled_tests2": "ENT archive II",
    "amplitude_tests": "AMP archive I",
    "amplitude_tests2": "AMP archive II",
    "amplitude_tests3": "AMP archive III",
    # Extended StatTests archive (channel-pair-resolved QEKG runs).
    "test_stat1+2": "QEKG Ch1+2 (I)",
    "test_stat1+2-2": "QEKG Ch1+2 (II)",
    "test_stat1+2-3": "QEKG Ch1+2 (III)",
    "test_stat1+3": "QEKG Ch1+3 (I)",
    "test_stat1+3-2": "QEKG Ch1+3 (II)",
    "tests1+3": "QEKG Ch1+3 (III)",
}

# Error-code strings that count as "clean" rows.  The cleaned archives
# use "00", but we accept any common synonym for a no-error code so
# that re-exports from other tools still work.
CLEAN_ERROR_CODES = {"", "0", "0.0", "00"}

# Default model constants from the bundled paper PDF (App. A,
# "Framework constants from the archive").
DEFAULT_C = 0.83          # finite-statistics coefficient
DEFAULT_MEDIAN_FLOOR = 7e-3
DEFAULT_BEST_FLOOR = 2e-3


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DatasetCurve:
    """All per-dataset summary numbers needed for plotting + reporting."""

    name: str
    source: Path
    clean_batches: int
    total_bits: int
    # Per-batch empirical eps = sqrt((1 - H1) * ln2 / 2).
    batch_epsilons: list[float] = field(default_factory=list)
    # Per-batch (1 - H1) defect; kept around so we can re-aggregate.
    batch_defects: list[float] = field(default_factory=list)
    # Running-prefix empirical curve (legacy).  cumulative_bits[i] is
    # the total number of clean bits seen after batch i, and
    # cumulative_epsilons[i] is the corresponding Pinsker bound.
    cumulative_bits: list[int] = field(default_factory=list)
    cumulative_epsilons: list[float] = field(default_factory=list)
    # Binned empirical curve (one point per bin size).
    binned_bits: list[int] = field(default_factory=list)
    binned_epsilon_mean: list[float] = field(default_factory=list)
    binned_epsilon_low: list[float] = field(default_factory=list)
    binned_epsilon_high: list[float] = field(default_factory=list)
    # Per-bin 95th-percentile epsilon -- the empirical upper envelope
    # that should track the model's c/sqrt(N) shape (high at low N,
    # decaying toward eta_emp as N grows).
    binned_epsilon_p95: list[float] = field(default_factory=list)

    @property
    def final_epsilon(self) -> float:
        return self.cumulative_epsilons[-1]

    @property
    def display_name(self) -> str:
        return DISPLAY_NAMES.get(self.name, self.name.replace("_", " "))

    @property
    def min_batch_epsilon(self) -> float:
        return min(self.batch_epsilons)

    @property
    def median_batch_epsilon(self) -> float:
        ordered = sorted(self.batch_epsilons)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        return 0.5 * (ordered[middle - 1] + ordered[middle])

    @property
    def asymptotic_floor(self) -> float:
        """Empirical eta_floor: epsilon from H1 of all clean bits merged.

        This is the right-hand asymptote of the running-prefix curve and
        the best estimate of the device's true bias floor that we can
        extract from this archive.
        """
        if not self.batch_defects:
            return 0.0
        mean_defect = sum(self.batch_defects) / len(self.batch_defects)
        return math.sqrt(mean_defect * LN2 / 2.0)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Build the CLI.

    Two styles are supported:

    1. ``--data path/to/file.csv`` (repeatable): explicit list of CSV
       paths.  Recommended for one-off runs.
    2. ``--data-dir DIR --datasets file1.csv file2.csv``: legacy
       directory + filename interface, used by older drivers.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- input -----------------------------------------------------------
    parser.add_argument(
        "--data",
        action="append",
        type=Path,
        default=None,
        metavar="CSV",
        help=(
            "Path to a CSV file with H1 / Error columns.  May be passed "
            "multiple times to overlay several archives."
        ),
    )
    parser.add_argument(
        "--group",
        action="append",
        default=None,
        metavar="LABEL=path1[,path2,...]",
        help=(
            "Merge multiple CSVs into one labelled empirical curve.  "
            "Repeatable.  Example: "
            "--group 'QEKG Ch1+2=stats/test_stat1+2.csv,stats/test_stat1+2-2.csv'.  "
            "Mutually exclusive with --data; use one or the other."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=(
            "Directory containing the archived CSV summaries.  Only used "
            "when --data is not given."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help=(
            "CSV file names inside --data-dir to load.  Ignored when "
            "--data is provided."
        ),
    )
    parser.add_argument(
        "--batch-bits",
        type=int,
        default=DEFAULT_BATCH_BITS,
        help="Number of bits summarized by each CSV row (default: 20000).",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Cap each input CSV at its first N clean batches; useful for "
            "aligning archives of different sizes.  Archives shorter "
            "than N are kept as-is (no padding).  Default: no cap."
        ),
    )

    # --- model constants -------------------------------------------------
    parser.add_argument(
        "--curve-c",
        type=float,
        default=DEFAULT_C,
        help=(
            "Finite-statistics coefficient c in eta_floor + c/sqrt(N).  "
            "Paper value: sqrt(2 ln 4)/2 ~ 0.83."
        ),
    )
    parser.add_argument(
        "--median-floor",
        type=float,
        default=DEFAULT_MEDIAN_FLOOR,
        help="Median eta_floor used for the paper's calibration curve.",
    )
    parser.add_argument(
        "--best-floor",
        type=float,
        default=DEFAULT_BEST_FLOOR,
        help="Best-run eta_floor used for the paper's calibration curve.",
    )

    # --- output ----------------------------------------------------------
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Single output image path.  The extension chooses the format "
            "(png, pdf, svg, ...).  Mutually exclusive with "
            "--output-prefix."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=IMAGES_DIR / "qrng_calibration_from_data",
        help=(
            "Output path prefix.  When --output is not given, the script "
            "writes both PREFIX.png and PREFIX.pdf."
        ),
    )

    # --- presentation ----------------------------------------------------
    parser.add_argument(
        "--title",
        default="QRNG Calibration From Archived Entropy Batches",
        help="Figure title.",
    )
    parser.add_argument(
        "--show-prefix-curve",
        action="store_true",
        help=(
            "Also plot the legacy running-prefix curve.  Off by default "
            "because it depends on batch order in the file and is "
            "misleading at small N."
        ),
    )
    parser.add_argument(
        "--show-upper-envelope",
        action="store_true",
        help=(
            "Overlay the per-bin 95th-percentile of epsilon.  This is "
            "the empirical quantity that should track the model's "
            "c/sqrt(N) shape: high at low N, decaying toward eta_emp."
        ),
    )
    parser.add_argument(
        "--hide-scatter",
        action="store_true",
        help="Suppress the per-batch scatter overlay.",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------


def is_clean_row(row: dict[str, str]) -> bool:
    """Return True if the row has no run-error code."""
    return row.get("Error", "").strip() in CLEAN_ERROR_CODES


def epsilon_from_h1(h1: float) -> float:
    """Pinsker-type empirical bias bound (paper App. A, eq. H1bias-app).

    Given a Shannon entropy ``h1`` (bits per symbol; 1.0 = ideal
    Bernoulli(1/2)), the worst-case binary-bias of the source is bounded
    by

        epsilon <= sqrt((1 - h1) * ln 2 / 2).

    We clamp negative defects (which can arise from numerical noise in
    the entropy estimator) to zero.
    """
    defect = max(0.0, 1.0 - h1)
    return math.sqrt(defect * LN2 / 2.0)


def _bin_indices(num_batches: int, bin_size: int) -> list[tuple[int, int]]:
    """Return non-overlapping (start, stop) slices that cover the data.

    The last partial bin (with fewer than ``bin_size`` batches) is
    dropped so every reported point represents the same N.
    """
    bins: list[tuple[int, int]] = []
    start = 0
    while start + bin_size <= num_batches:
        bins.append((start, start + bin_size))
        start += bin_size
    return bins


def _percentile(values: Sequence[float], pct: float) -> float:
    """Pure-Python linear-interpolation percentile (numpy-free).

    ``pct`` is in [0, 100].  For very small samples the result is
    interpolated between the two neighbouring order statistics.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (rank - lo) * (ordered[hi] - ordered[lo])


def _build_binned_curve(
    defects: Sequence[float],
    batch_bits: int,
    num_points: int = 40,
) -> tuple[list[int], list[float], list[float], list[float], list[float]]:
    """Build the binned empirical eps(N) curve.

    For each target N (log-spaced from one batch to all batches) we
    compute k = max(1, round(N / batch_bits)) and split the dataset
    into bins of k batches.  Each bin yields an epsilon via the
    Pinsker formula applied to the bin's mean (1 - H1).  For each bin
    size we report:

      * mean across bins (the typical-case observation),
      * +/- 1 sigma band (the spread),
      * 95th percentile across bins (the empirical upper envelope; this
        is the curve that should track the model's c/sqrt(N) shape --
        high at low N, decaying to eta_emp as N grows).
    """
    num_batches = len(defects)
    if num_batches == 0:
        return [], [], [], [], []

    # Pick bin sizes (in batches) log-uniformly from 1 to num_batches.
    if num_batches == 1:
        ks = [1]
    else:
        log_lo, log_hi = math.log(1.0), math.log(float(num_batches))
        # Deduplicate while preserving order.
        raw = (
            round(math.exp(log_lo + index * (log_hi - log_lo) / (num_points - 1)))
            for index in range(num_points)
        )
        ks = sorted(set(int(max(1, k)) for k in raw))

    sample_counts: list[int] = []
    eps_mean: list[float] = []
    eps_low: list[float] = []
    eps_high: list[float] = []
    eps_p95: list[float] = []

    for k in ks:
        bins = _bin_indices(num_batches, k)
        if not bins:
            continue
        # Per-bin epsilon = Pinsker on the bin's mean defect.
        bin_eps: list[float] = []
        for start, stop in bins:
            mean_defect = sum(defects[start:stop]) / (stop - start)
            bin_eps.append(math.sqrt(mean_defect * LN2 / 2.0))

        sample_counts.append(k * batch_bits)
        mean = sum(bin_eps) / len(bin_eps)
        if len(bin_eps) > 1:
            variance = sum((e - mean) ** 2 for e in bin_eps) / (len(bin_eps) - 1)
            sigma = math.sqrt(max(0.0, variance))
        else:
            sigma = 0.0
        eps_mean.append(mean)
        eps_low.append(max(0.0, mean - sigma))
        eps_high.append(mean + sigma)
        eps_p95.append(_percentile(bin_eps, 95.0))

    return sample_counts, eps_mean, eps_low, eps_high, eps_p95


def _read_clean_h1(path: Path, max_batches: int | None = None) -> list[float]:
    """Return the H1 column for clean rows of ``path``.

    If ``max_batches`` is given and positive, stop after collecting that
    many clean rows (archives shorter than the cap are returned in
    full -- no padding).
    """
    h1s: list[float] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not is_clean_row(row):
                continue
            try:
                h1s.append(float(row["H1"]))
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"Could not parse H1 from row in {path}: {row!r}"
                ) from exc
            if max_batches is not None and max_batches > 0 and len(h1s) >= max_batches:
                break
    return h1s


def load_dataset(
    paths: Path | list[Path],
    batch_bits: int,
    label: str | None = None,
    max_batches: int | None = None,
) -> DatasetCurve:
    """Read one or more CSV archives and pre-compute everything we need.

    ``paths`` may be a single ``Path`` or a list of ``Path``s; in the
    list case the per-batch H1 streams are concatenated in the given
    order (so the curve is computed over the union of the archives).
    ``label`` overrides the default display name; if not given the
    label is derived from the first path's stem.
    ``max_batches``, if given, caps each input CSV at its first N clean
    batches before concatenation; this is the right semantics for
    "align archive lengths for a fair comparison" without padding.
    """
    if isinstance(paths, Path):
        path_list: list[Path] = [paths]
    else:
        path_list = list(paths)
    if not path_list:
        raise ValueError("load_dataset requires at least one path")

    batch_defects: list[float] = []
    batch_epsilons: list[float] = []
    cumulative_bits: list[int] = []
    cumulative_epsilons: list[float] = []
    cumulative_defect = 0.0

    for path in path_list:
        for h1 in _read_clean_h1(path, max_batches=max_batches):
            # --- per-batch Pinsker bound -------------------------------
            defect = max(0.0, 1.0 - h1)
            batch_defects.append(defect)
            batch_epsilons.append(math.sqrt(defect * LN2 / 2.0))

            # --- running-prefix cumulative bound -----------------------
            # cumulative_defect / k is the mean of (1 - H1) over the
            # first k batches; by Pinsker that gives a (slightly
            # conservative) estimate of the bias bound from the
            # concatenated data.
            cumulative_defect += defect
            k = len(batch_epsilons)
            mean_defect = cumulative_defect / k
            cumulative_bits.append(k * batch_bits)
            cumulative_epsilons.append(math.sqrt(mean_defect * LN2 / 2.0))

    if not batch_epsilons:
        raise ValueError(f"No clean rows found in {[str(p) for p in path_list]}")

    # --- binned eps(N) curve (the meaningful comparison to the model)
    binned_bits, binned_mean, binned_low, binned_high, binned_p95 = (
        _build_binned_curve(batch_defects, batch_bits)
    )

    name = label if label is not None else path_list[0].stem
    return DatasetCurve(
        name=name,
        source=path_list[0],  # representative source path; full list available via batch counts
        clean_batches=len(batch_epsilons),
        total_bits=len(batch_epsilons) * batch_bits,
        batch_epsilons=batch_epsilons,
        batch_defects=batch_defects,
        cumulative_bits=cumulative_bits,
        cumulative_epsilons=cumulative_epsilons,
        binned_bits=binned_bits,
        binned_epsilon_mean=binned_mean,
        binned_epsilon_low=binned_low,
        binned_epsilon_high=binned_high,
        binned_epsilon_p95=binned_p95,
    )


def logspace(start: float, stop: float, points: int) -> list[float]:
    """Pure-Python log-spaced sample of `points` values in [start, stop]."""
    if start <= 0 or stop <= 0:
        raise ValueError("logspace endpoints must be positive")
    if points < 2:
        return [start]
    log_start = math.log(start)
    log_stop = math.log(stop)
    return [
        math.exp(log_start + index * (log_stop - log_start) / (points - 1))
        for index in range(points)
    ]


def model_curve(
    sample_counts: Iterable[float], eta_floor: float, c_value: float
) -> list[float]:
    """Evaluate the paper's bound eta_floor + c/sqrt(N) at each N."""
    return [eta_floor + c_value / math.sqrt(n) for n in sample_counts]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _resolve_inputs(args: argparse.Namespace) -> list[tuple[str | None, list[Path]]]:
    """Turn the CLI arguments into a list of (label, paths) groups.

    Each entry of the returned list becomes one ``DatasetCurve``.  In
    the single-file paths (``--data`` or ``--datasets``) the label is
    ``None`` so ``load_dataset`` uses the CSV stem.  When ``--group`` is
    used the explicit label is preserved.
    """
    if args.group:
        groups: list[tuple[str | None, list[Path]]] = []
        for spec in args.group:
            if "=" not in spec:
                raise ValueError(
                    f"--group expects 'LABEL=path[,path,...]', got {spec!r}"
                )
            label, files = spec.split("=", 1)
            paths = [Path(p.strip()) for p in files.split(",") if p.strip()]
            if not paths:
                raise ValueError(f"--group {label!r} has no paths")
            groups.append((label.strip() or None, paths))
        return groups
    if args.data:
        return [(None, [p]) for p in args.data]
    return [(None, [args.data_dir / name]) for name in args.datasets]


def _resolve_outputs(args: argparse.Namespace) -> list[Path]:
    """Return one or more output paths."""
    if args.output is not None:
        return [args.output]
    return [
        args.output_prefix.with_suffix(".png"),
        args.output_prefix.with_suffix(".pdf"),
    ]


def plot_curves(curves: list[DatasetCurve], args: argparse.Namespace) -> list[Path]:
    """Render the calibration figure and write it to one or more files."""
    output_paths = _resolve_outputs(args)
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    # ----- x-axis range: span all empirical N values --------------------
    max_bits = max(curve.total_bits for curve in curves)
    # The smallest N we plot is one batch; the model curve is evaluated
    # over the full visible range.
    min_bits = args.batch_bits
    model_samples = logspace(float(min_bits), float(max_bits), 400)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)

    # ----- 1. Paper's model upper bounds (red solid, blue dashed) ------
    model_specs = [
        (args.median_floor, "Median model (paper)", "#b6412c", "-"),
        (args.best_floor, "Best-run model (paper)", "#3b6da8", "--"),
    ]
    for eta_floor, label, color, linestyle in model_specs:
        ax.plot(
            model_samples,
            model_curve(model_samples, eta_floor, args.curve_c),
            color=color,
            linestyle=linestyle,
            linewidth=2.2,
            label=f"{label}: eta={eta_floor:.3g}, c={args.curve_c:.2f}",
            zorder=3,
        )

    # ----- 2. Empirical binned eps(N) curves ----------------------------
    empirical_colors = ["#222222", "#3f7f5f", "#6a4c93", "#8b5e34", "#9b1d20"]
    for index, curve in enumerate(curves):
        color = empirical_colors[index % len(empirical_colors)]

        # Per-batch scatter (optional, on by default).
        if not args.hide_scatter:
            batch_n = [args.batch_bits] * len(curve.batch_epsilons)
            ax.scatter(
                batch_n,
                curve.batch_epsilons,
                color=color,
                s=4,
                alpha=0.06,
                edgecolors="none",
                zorder=1,
            )

        # Binned mean with +/- 1 sigma band.  This is the right
        # empirical analogue of the model bound.
        ax.fill_between(
            curve.binned_bits,
            curve.binned_epsilon_low,
            curve.binned_epsilon_high,
            color=color,
            alpha=0.18,
            linewidth=0,
            zorder=2,
        )
        ax.plot(
            curve.binned_bits,
            curve.binned_epsilon_mean,
            color=color,
            linewidth=2.0,
            label=(
                f"Empirical {curve.display_name}: eta_emp="
                f"{curve.asymptotic_floor:.4f}, batches={curve.clean_batches}"
            ),
            zorder=4,
        )

        # Mark the asymptote (eta_emp) as a small horizontal dashed line
        # near the right edge.
        ax.hlines(
            curve.asymptotic_floor,
            xmin=max_bits / 5.0,
            xmax=max_bits,
            colors=color,
            linestyles=":",
            linewidth=1.0,
            alpha=0.9,
            zorder=3,
        )

        # Final cumulative point (matches the legacy curve's endpoint).
        ax.scatter(
            [curve.cumulative_bits[-1]],
            [curve.cumulative_epsilons[-1]],
            color=color,
            s=34,
            zorder=5,
        )

        # Legacy running-prefix curve, only if requested.
        if args.show_prefix_curve:
            ax.plot(
                curve.cumulative_bits,
                curve.cumulative_epsilons,
                color=color,
                linewidth=1.0,
                alpha=0.55,
                linestyle="--",
                label=f"Running prefix {curve.display_name}",
                zorder=2,
            )

        # Empirical upper envelope (per-bin 95th percentile).  This is
        # the empirical quantity that should track the model's
        # c/sqrt(N) shape: high at low N (few bits per bin, large
        # statistical fluctuation), decaying toward eta_emp at large N.
        if args.show_upper_envelope:
            ax.plot(
                curve.binned_bits,
                curve.binned_epsilon_p95,
                color=color,
                linewidth=1.4,
                alpha=0.85,
                linestyle="-.",
                label=f"95th pct (per bin) {curve.display_name}",
                zorder=4,
            )

    # ----- 3. Axes, legend, annotation ----------------------------------
    ax.set_xscale("log")
    ax.set_xlabel("Sample count N (bits)")
    ax.set_ylabel(r"Bias bound $\varepsilon$")
    ax.set_title(args.title)

    # Pick a reasonable y-range: enough headroom for the model curves'
    # left tail but not so much that the data is squashed.
    y_top = max(
        args.median_floor + args.curve_c / math.sqrt(min_bits),
        max(max(curve.binned_epsilon_high) for curve in curves) * 1.4,
    )
    ax.set_ylim(0.0, 1.05 * y_top)
    ax.grid(True, which="both", linewidth=0.5, alpha=0.35)
    ax.legend(loc="upper right", fontsize=9, frameon=True)

    note = (
        "Empirical lines: mean per-bin eps with +/-1 sigma band\n"
        "(bins are non-overlapping groups of N/20000 batches).\n"
        "Dotted horizontals: empirical eta_emp from all clean bits merged.\n"
        "Model curves: eta_floor + c/sqrt(N) -- upper bounds, not fits."
    )
    ax.text(
        0.015,
        0.02,
        note,
        transform=ax.transAxes,
        fontsize=8.5,
        va="bottom",
        ha="left",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "alpha": 0.92,
            "edgecolor": "#cccccc",
        },
    )

    written: list[Path] = []
    for path in output_paths:
        # Pick a reasonable DPI for raster outputs; vector formats ignore it.
        fig.savefig(path, dpi=220)
        written.append(path)
    plt.close(fig)
    return written


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_summary(curves: list[DatasetCurve], outputs: list[Path]) -> None:
    """Echo the dataset summary numbers to stdout for sanity checking."""
    print("Generated calibration plot:")
    for path in outputs:
        print(f"  {path.suffix.lstrip('.').upper():>4}: {path}")
    for curve in curves:
        print(
            f"  {curve.name}: clean_batches={curve.clean_batches}, "
            f"total_bits={curve.total_bits}, "
            f"median_batch_eps={curve.median_batch_epsilon:.6f}, "
            f"min_batch_eps={curve.min_batch_epsilon:.6f}, "
            f"eta_emp={curve.asymptotic_floor:.6f}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    inputs = _resolve_inputs(args)

    curves = [
        load_dataset(paths, args.batch_bits, label=label, max_batches=args.max_batches)
        for label, paths in inputs
    ]
    outputs = plot_curves(curves, args)
    print_summary(curves, outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
