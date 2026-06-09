#!/usr/bin/env python3
"""Generate the AIS31/NIST randomness-test panels for the QRNG paper.

Produces two PNG/PDF figures comparing the lab QRNG to a vetted
classical CSPRNG baseline drawn from Python's ``secrets`` module
(which on POSIX delegates to ``/dev/urandom`` and on macOS to the
system CSPRNG).  The two figures mirror the layout of the legacy
``EntropyTest.png`` / ``PValueTest.png`` panels referenced in the
paper, but are regenerated from the *same* CSVs that drive
Figure 5 (entropy-derived calibration curve).

Per-batch statistics shown
--------------------------
For each 20,000-bit batch we recompute the standard quantities used
by AIS31 / NIST SP 800-22:

* ``H1``  Shannon entropy of the empirical bit-frequency, in bits
          per symbol.  Indistinguishable from 1.0 means the batch
          passes the entropy estimate.
* ``T1``  Monobit count (number of ones in the batch).  Under H0
          this is Binomial(20000, 1/2): mean 10000, std ~ 70.7.

The QRNG values come directly from the CSV columns ``H1`` and ``T1``;
the PRNG baseline is computed by drawing the same number of bits
from ``secrets.token_bytes`` and applying the same per-batch reduction.

Usage
-----
::

    python experiments/plot_randomness_tests.py
    python experiments/plot_randomness_tests.py \
        --data experiments/experiment_data/entangled_tests.csv \
        --output-entropy experiments/Images/EntropyTest.png \
        --output-monobit experiments/Images/MonobitTest.png

Both ``--output-*`` flags choose the format from the extension (PNG
default, PDF works too).  Pass ``--seed`` to make the PRNG baseline
deterministic for reproducibility checks.
"""

from __future__ import annotations

import argparse
import csv
import math
import secrets
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
STATTEST_DIR = SCRIPT_DIR / "QEKG" / "TestData" / "StatTests"
IMAGES_DIR = SCRIPT_DIR / "Images"

LN2 = math.log(2.0)
DEFAULT_BATCH_BITS = 20_000
DEFAULT_DATASETS = ("test_stat1+2.csv", "test_stat1+2-2.csv", "test_stat1+2-3.csv")
DEFAULT_DATA_DIR = STATTEST_DIR
DEFAULT_OUTPUT_ENTROPY = IMAGES_DIR / "EntropyTest.png"
DEFAULT_OUTPUT_MONOBIT = IMAGES_DIR / "MonobitTest.png"
CLEAN_ERROR_CODES = {"", "0", "0.0", "00"}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def shannon_entropy_from_ones(ones: int, total: int) -> float:
    """Shannon entropy (bits/symbol) of a Bernoulli(p) source with
    ``p = ones / total``.  Returns 1.0 when ``ones == total/2``.
    """
    if total == 0:
        return 0.0
    p = ones / total
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def prng_baseline(num_batches: int, batch_bits: int, seed: int | None) -> tuple[list[float], list[int]]:
    """Draw ``num_batches * batch_bits`` random bits from Python's
    ``secrets`` (or a seeded ``random`` if ``seed`` is given for
    reproducibility) and return the per-batch (H1, monobit count).
    """
    h1: list[float] = []
    t1: list[int] = []

    if seed is None:
        # Cryptographically-secure, OS-vetted CSPRNG.  No state to
        # carry across batches.
        def draw_bytes(n: int) -> bytes:
            return secrets.token_bytes(n)
    else:
        # Seeded path for deterministic regeneration; uses NumPy-free
        # Python ``random`` (Mersenne Twister) but we apply it
        # byte-wise to keep the comparison apples-to-apples.
        import random

        rng = random.Random(seed)

        def draw_bytes(n: int) -> bytes:
            return rng.randbytes(n)

    byte_count = batch_bits // 8
    if batch_bits % 8 != 0:
        raise ValueError("batch_bits must be a multiple of 8")

    for _ in range(num_batches):
        chunk = draw_bytes(byte_count)
        ones = sum(bin(b).count("1") for b in chunk)
        h1.append(shannon_entropy_from_ones(ones, batch_bits))
        t1.append(ones)

    return h1, t1


def load_qrng(
    path: Path, max_batches: int | None = None
) -> tuple[list[float], list[int]]:
    """Return ``(H1, T1)`` lists for all clean rows of ``path``.

    If ``max_batches`` is given and positive, stop after collecting that
    many clean rows (archives shorter than the cap are returned in
    full -- no padding).
    """
    h1: list[float] = []
    t1: list[int] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("Error", "").strip() not in CLEAN_ERROR_CODES:
                continue
            try:
                h1.append(float(row["H1"]))
                # T1 is the monobit count of the 20 000-bit batch.
                t1.append(int(round(float(row["T1"]))))
            except (KeyError, ValueError):
                continue
            if max_batches is not None and max_batches > 0 and len(h1) >= max_batches:
                break
    if not h1:
        raise ValueError(f"No clean rows found in {path}")
    return h1, t1


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _scatter_panel(
    ax,
    series: list[tuple[list[float], str, str]],
    *,
    title: str,
    ylabel: str,
    ymin: float | None,
    ymax: float | None,
    hline: float | None = None,
    hline_label: str | None = None,
    sigma_band: tuple[float, float] | None = None,
) -> None:
    """One scatter panel.

    ``series`` is a list of ``(values, label, color)`` tuples; each is
    plotted as its own scatter so per-archive splits stay visible.

    ``sigma_band``, if given as ``(mean, sigma)``, shades [mean-sigma,
    mean+sigma] as a soft grey band; this gives the eye a reference for
    the expected dispersion under H0.
    """
    # Soft +/-1 sigma band behind the points so the H0 spread is
    # visually obvious (avoids the "looks off-center" illusion you get
    # from alpha-blended scatter alone).
    if sigma_band is not None:
        mean, sigma = sigma_band
        ax.axhspan(mean - sigma, mean + sigma, color="#bbbbbb", alpha=0.18, zorder=0)

    x_offset = 0
    for values, label, color in series:
        xs = list(range(x_offset + 1, x_offset + len(values) + 1))
        ax.scatter(
            xs,
            values,
            s=4,
            alpha=0.28,
            color=color,
            edgecolors="none",
            label=label,
            zorder=2,
        )
        x_offset += len(values)

    ax.set_title(title, fontfamily="monospace")
    ax.set_xlabel("Batch index")
    ax.set_ylabel(ylabel)
    if ymin is not None or ymax is not None:
        ax.set_ylim(ymin, ymax)

    if hline is not None:
        ax.axhline(
            hline,
            color="#222222",
            linestyle="--",
            linewidth=1.0,
            label=hline_label,
            zorder=3,
        )

    # Build legend with non-transparent scatter handles so the colour
    # swatches read clearly even though the data alpha is low.
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            loc="lower right",
            fontsize=8,
            framealpha=0.92,
            markerscale=2.0,
            scatterpoints=1,
        )
    ax.grid(True, linewidth=0.4, alpha=0.35)


def plot_two_panels(
    prng_series,
    qrng_series,
    *,
    output: Path,
    panel_title_left: str,
    panel_title_right: str,
    ylabel: str,
    ymin: float | None = None,
    ymax: float | None = None,
    hline: float | None = None,
    hline_label: str | None = None,
    sigma_band: tuple[float, float] | None = None,
    figure_title: str | None = None,
) -> Path:
    """Render PRNG (left) and QRNG (right) scatter panels.

    Each ``*_series`` is a list of ``(values, label, color)`` tuples so
    we can split the QRNG into its two campaigns and keep the join
    explicit rather than misleading.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True, constrained_layout=True)
    _scatter_panel(
        axes[0],
        prng_series,
        title=panel_title_left,
        ylabel=ylabel,
        ymin=ymin,
        ymax=ymax,
        hline=hline,
        hline_label=hline_label,
        sigma_band=sigma_band,
    )
    _scatter_panel(
        axes[1],
        qrng_series,
        title=panel_title_right,
        ylabel="",
        ymin=ymin,
        ymax=ymax,
        hline=hline,
        hline_label=hline_label,
        sigma_band=sigma_band,
    )
    if figure_title:
        fig.suptitle(figure_title, fontsize=11)
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data",
        action="append",
        type=Path,
        default=None,
        help="Path to a CSV with H1/T1/Error columns. Repeatable; archives are concatenated.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory used together with --datasets when --data is not given.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="CSV filenames in --data-dir.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Cap each input CSV at its first N clean batches; useful for "
            "aligning archives of different sizes.  Archives shorter "
            "than N are kept as-is (no padding).  The PRNG baseline is "
            "drawn to match the resulting summed QRNG batch count. "
            "Default: no cap."
        ),
    )
    parser.add_argument(
        "--batch-bits",
        type=int,
        default=DEFAULT_BATCH_BITS,
        help="Bits per batch (default 20000, matches the Mathematica notebooks).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional integer seed for the PRNG baseline (deterministic). Default uses secrets.",
    )
    parser.add_argument(
        "--output-entropy",
        type=Path,
        default=DEFAULT_OUTPUT_ENTROPY,
        help="Output path for the H1 two-panel figure.",
    )
    parser.add_argument(
        "--output-monobit",
        type=Path,
        default=DEFAULT_OUTPUT_MONOBIT,
        help="Output path for the monobit (T1) two-panel figure.",
    )
    return parser.parse_args()


# Per-archive colours for the QRNG scatter (kept distinct so the
# multi-campaign nature of the archive is unmistakable).
_QRNG_COLORS = ["#1f4e8a", "#b6412c", "#3f7f5f", "#6a4c93", "#8b5e34"]
# Single neutral colour for the PRNG baseline.
_PRNG_COLOR = "#2a4d8f"

# Friendly labels for the per-archive legends; mirrors the
# calibration script.
DISPLAY_NAMES = {
    "entangled_tests": "ENT archive I",
    "entangled_tests2": "ENT archive II",
    "amplitude_tests": "AMP archive I",
    "amplitude_tests2": "AMP archive II",
    "amplitude_tests3": "AMP archive III",
}


def main() -> int:
    args = parse_args()

    # --- Load QRNG --------------------------------------------------------
    if args.data:
        paths = list(args.data)
    else:
        paths = [args.data_dir / n for n in args.datasets]

    # Keep per-archive arrays so we can plot them as separate series
    # and the multi-campaign join is visible as an intentional colour
    # split rather than a mysterious step.
    qrng_per_archive: list[tuple[Path, list[float], list[int]]] = []
    for p in paths:
        h1, t1 = load_qrng(p, max_batches=args.max_batches)
        qrng_per_archive.append((p, h1, t1))
    n_batches = sum(len(h1) for _, h1, _ in qrng_per_archive)
    cap_note = ""
    if args.max_batches is not None and args.max_batches > 0:
        capped = sum(1 for _, h1, _ in qrng_per_archive if len(h1) == args.max_batches)
        cap_note = (
            f" (--max-batches {args.max_batches}: {capped}/{len(paths)} "
            "archive(s) truncated to the cap)"
        )
    print(
        f"QRNG: {n_batches} clean batches loaded from {len(paths)} CSV file(s){cap_note}."
    )
    for p, h1, _ in qrng_per_archive:
        print(f"  {p.name}: {len(h1)} clean batches")

    # --- PRNG baseline ----------------------------------------------------
    print(
        f"PRNG: drawing {n_batches} batches of {args.batch_bits} bits from "
        f"{'random.Random(seed=' + str(args.seed) + ')' if args.seed is not None else 'secrets.token_bytes (OS CSPRNG)'}..."
    )
    h1_p, t1_p = prng_baseline(n_batches, args.batch_bits, args.seed)

    # --- Series builders --------------------------------------------------
    prng_h1_series = [(h1_p, "PC CSPRNG baseline", _PRNG_COLOR)]
    prng_t1_series = [([float(x) for x in t1_p], "PC CSPRNG baseline", _PRNG_COLOR)]

    qrng_h1_series = []
    qrng_t1_series = []
    for index, (p, h1, t1) in enumerate(qrng_per_archive):
        color = _QRNG_COLORS[index % len(_QRNG_COLORS)]
        label = DISPLAY_NAMES.get(p.stem, p.stem.replace("_", " "))
        qrng_h1_series.append((h1, label, color))
        qrng_t1_series.append(([float(x) for x in t1], label, color))

    # --- Figure 1: entropy scatter ---------------------------------------
    # Sigma band for H1 under H0(unbiased binary, N=batch_bits) using
    # the plug-in estimator's first-order bias 1/(2 N ln 2):
    h1_floor = 1.0 - 1.0 / (2.0 * args.batch_bits * LN2)
    h1_sigma = math.sqrt(1.0 / (4.0 * args.batch_bits))  # rough scale
    plot_two_panels(
        prng_h1_series,
        qrng_h1_series,
        output=args.output_entropy,
        panel_title_left=f"Entropy: PC CSPRNG baseline  (n={n_batches})",
        panel_title_right=f"Entropy: lab QRNG  (n={n_batches})",
        ylabel="$H_1$  (bits/symbol)",
        ymin=0.99970,
        ymax=1.00001,
        hline=1.0,
        hline_label=r"$H_1=1$ (ideal)",
        sigma_band=(h1_floor, h1_sigma),
    )
    print(f"  wrote {args.output_entropy}")

    # --- Figure 2: monobit scatter ---------------------------------------
    expected = args.batch_bits / 2
    sigma = math.sqrt(args.batch_bits / 4)
    plot_two_panels(
        prng_t1_series,
        qrng_t1_series,
        output=args.output_monobit,
        panel_title_left=f"Monobit: PC CSPRNG baseline  (n={n_batches})",
        panel_title_right=f"Monobit: lab QRNG  (n={n_batches})",
        ylabel=f"# ones in {args.batch_bits}-bit batch",
        ymin=expected - 4.5 * sigma,
        ymax=expected + 4.5 * sigma,
        hline=expected,
        hline_label=f"$H_0$ mean = {int(expected)}  ($\\sigma\\approx${sigma:.1f})",
        sigma_band=(expected, sigma),
    )
    print(f"  wrote {args.output_monobit}")

    # --- Quick summary ----------------------------------------------------
    def stats(xs):
        s = sorted(xs)
        mean = sum(s) / len(s)
        return mean, s[len(s) // 4], s[len(s) // 2], s[3 * len(s) // 4]

    qrng_h1_all = [x for _, h, _ in qrng_per_archive for x in h]
    qrng_t1_all = [x for _, _, t in qrng_per_archive for x in t]

    m, p25, p50, p75 = stats(qrng_h1_all)
    print(f"  QRNG H1:  mean={m:.7f}  25%={p25:.7f}  50%={p50:.7f}  75%={p75:.7f}")
    m, p25, p50, p75 = stats(h1_p)
    print(f"  PRNG H1:  mean={m:.7f}  25%={p25:.7f}  50%={p50:.7f}  75%={p75:.7f}")
    m, p25, p50, p75 = stats(qrng_t1_all)
    print(f"  QRNG T1:  mean={m:.2f}     25%={p25}     50%={p50}     75%={p75}")
    m, p25, p50, p75 = stats(t1_p)
    print(f"  PRNG T1:  mean={m:.2f}     25%={p25}     50%={p50}     75%={p75}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
