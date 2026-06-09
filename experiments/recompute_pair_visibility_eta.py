#!/usr/bin/env python3
"""Recompute the detector-pair visibility split and QRNG eta_emp split.

This script is a reproducible companion to the Appendix-A discussion in
the paper.  It pulls the two quantitative ingredients of the claim from
the archived CSVs already used elsewhere in the repository:

1. Malus-fit fringe visibilities from the BBO waveplate-rotation scans
   (raw coincidence counts, one file per outer stage).
2. Empirical QRNG bias floors eta_emp from the channel-pair-resolved
   StatTests archives via the same Pinsker/H1 computation used by
   ``plot_qrng_calibration.py``.

The script intentionally does not invent any new processing pipeline.
It reuses the same fitting and eta_emp logic as the existing plotting
scripts so the published numbers can be regenerated from one command.

Default output includes:

* representative-stage visibilities for Ch1&3, Ch1&4, Ch2&3, Ch2&4,
* grouped visibility means for the detector-3 and detector-4 pairs,
* stagewise best-pair visibility mean / std,
* eta_emp for the merged Ch1+2 and Ch1+3 StatTests campaigns,
* the eta_emp difference between those two campaigns.

Usage
-----

    python experiments/recompute_pair_visibility_eta.py
    python experiments/recompute_pair_visibility_eta.py --json-out /tmp/pair_split.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plot_bbo_malus import COINCIDENCE_PAIRS, fit_malus, load_malus_scan
from plot_qrng_calibration import DEFAULT_BATCH_BITS, load_dataset


SCRIPT_DIR = Path(__file__).resolve().parent
TESTDATA_DIR = SCRIPT_DIR / "QEKG" / "TestData"
IMAGES_DIR = SCRIPT_DIR / "Images"

DEFAULT_BBO_DIR = TESTDATA_DIR / "BBO-QWP-ROTATION-TEST"
DEFAULT_BBO_GLOB = "experiment_rotation_02_*.csv"
DEFAULT_STATTEST_DIR = TESTDATA_DIR / "StatTests"
DEFAULT_CH12 = ("test_stat1+2.csv", "test_stat1+2-2.csv", "test_stat1+2-3.csv")
DEFAULT_CH13 = ("test_stat1+3.csv", "test_stat1+3-2.csv", "tests1+3.csv")
DEFAULT_PLOT_OUT = IMAGES_DIR / "eta_g_insensitivity.pdf"
DEFAULT_MEDIAN_FLOOR = 7e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bbo-dir",
        type=Path,
        default=DEFAULT_BBO_DIR,
        help="Directory containing BBO rotation CSVs.",
    )
    parser.add_argument(
        "--bbo-glob",
        default=DEFAULT_BBO_GLOB,
        help="Glob selecting the BBO rotation files inside --bbo-dir.",
    )
    parser.add_argument(
        "--representative-index",
        type=int,
        default=45,
        help="0-based stage index used for the representative-stage visibility summary.",
    )
    parser.add_argument(
        "--stattest-dir",
        type=Path,
        default=DEFAULT_STATTEST_DIR,
        help="Directory containing the StatTests CSVs.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional JSON output path for machine-readable summaries.",
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=DEFAULT_PLOT_OUT,
        help=(
            "Output path for the visibility-vs-eta_emp comparison figure. "
            "A sibling file with the other extension (.png/.pdf) is also written."
        ),
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip figure generation (print/JSON summaries only).",
    )
    parser.add_argument(
        "--median-floor",
        type=float,
        default=DEFAULT_MEDIAN_FLOOR,
        help="Median eta_floor reference line drawn on the eta_emp panel.",
    )
    return parser.parse_args()


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return variance**0.5


def _load_scans(data_dir: Path, glob: str) -> list:
    files = sorted(data_dir.glob(glob))
    if not files:
        raise FileNotFoundError(f"No files matched {glob!r} in {data_dir}")
    return [load_malus_scan(path, angle_start=0.0, angle_step=1.0) for path in files]


def _pair_stage_visibilities(scans: list) -> dict[str, list[float]]:
    per_pair: dict[str, list[float]] = {pair: [] for pair in COINCIDENCE_PAIRS}
    for scan in scans:
        for pair in COINCIDENCE_PAIRS:
            per_pair[pair].append(fit_malus(scan.thetas_rad, scan.counts[pair]).visibility)
    return per_pair


def _summarize_visibility(scans: list, representative_index: int) -> dict:
    if not (0 <= representative_index < len(scans)):
        raise IndexError(
            f"representative-index {representative_index} out of range for {len(scans)} scans"
        )

    rep = scans[representative_index]
    representative = {
        pair: fit_malus(rep.thetas_rad, rep.counts[pair]).visibility
        for pair in COINCIDENCE_PAIRS
    }

    by_stage = _pair_stage_visibilities(scans)

    detector3_pairs = ["Ch1&3", "Ch2&3"]
    detector4_pairs = ["Ch1&4", "Ch2&4"]
    det3_all = [value for pair in detector3_pairs for value in by_stage[pair]]
    det4_all = [value for pair in detector4_pairs for value in by_stage[pair]]
    best_per_stage = [max(by_stage[pair][index] for pair in COINCIDENCE_PAIRS) for index in range(len(scans))]

    return {
        "stage_count": len(scans),
        "representative_index": representative_index,
        "representative_fixed_settings": rep.fixed_settings,
        "representative_visibility": representative,
        "stagewise_visibility_mean": {pair: _mean(values) for pair, values in by_stage.items()},
        "stagewise_visibility_std": {pair: _sample_std(values) for pair, values in by_stage.items()},
        "grouped_visibility_mean": {
            "detector3_pairs": _mean(det3_all),
            "detector4_pairs": _mean(det4_all),
        },
        "grouped_visibility_std": {
            "detector3_pairs": _sample_std(det3_all),
            "detector4_pairs": _sample_std(det4_all),
        },
        "best_pair_per_stage_mean": _mean(best_per_stage),
        "best_pair_per_stage_std": _sample_std(best_per_stage),
    }


def _load_eta_summary(stattest_dir: Path, label: str, filenames: tuple[str, ...]) -> dict:
    curve = load_dataset(
        [stattest_dir / filename for filename in filenames],
        batch_bits=DEFAULT_BATCH_BITS,
        label=label,
    )
    return {
        "label": label,
        "files": list(filenames),
        "clean_batches": curve.clean_batches,
        "total_bits": curve.total_bits,
        "eta_emp": curve.asymptotic_floor,
    }


def _print_summary(visibility: dict, eta_ch12: dict, eta_ch13: dict) -> None:
    rep = visibility["representative_visibility"]
    print("Representative-stage Malus visibilities")
    print(f"  stage index: {visibility['representative_index']}")
    print(f"  fixed settings: {visibility['representative_fixed_settings']}")
    for pair in COINCIDENCE_PAIRS:
        print(f"  {pair}: V={rep[pair]:.6f}")
    print()

    print("Stagewise visibility summary")
    print(
        "  grouped detector-3 pairs (Ch1&3, Ch2&3): "
        f"mean={visibility['grouped_visibility_mean']['detector3_pairs']:.6f}, "
        f"std={visibility['grouped_visibility_std']['detector3_pairs']:.6f}"
    )
    print(
        "  grouped detector-4 pairs (Ch1&4, Ch2&4): "
        f"mean={visibility['grouped_visibility_mean']['detector4_pairs']:.6f}, "
        f"std={visibility['grouped_visibility_std']['detector4_pairs']:.6f}"
    )
    print(
        "  best pair per stage: "
        f"mean={visibility['best_pair_per_stage_mean']:.6f}, "
        f"std={visibility['best_pair_per_stage_std']:.6f}"
    )
    print()

    delta = abs(eta_ch12["eta_emp"] - eta_ch13["eta_emp"])
    print("QRNG eta_emp from merged StatTests campaigns")
    print(
        f"  {eta_ch12['label']}: eta_emp={eta_ch12['eta_emp']:.6f}, "
        f"batches={eta_ch12['clean_batches']}, bits={eta_ch12['total_bits']}"
    )
    print(
        f"  {eta_ch13['label']}: eta_emp={eta_ch13['eta_emp']:.6f}, "
        f"batches={eta_ch13['clean_batches']}, bits={eta_ch13['total_bits']}"
    )
    print(f"  |delta eta_emp|={delta:.6f}")


def _save_figure(fig, output: Path) -> list[Path]:
    """Save ``fig`` to ``output`` plus a sibling with the other vector/raster suffix."""
    output.parent.mkdir(parents=True, exist_ok=True)
    written = [output]
    fig.savefig(output, dpi=200)
    sibling_suffix = ".png" if output.suffix.lower() == ".pdf" else ".pdf"
    sibling = output.with_suffix(sibling_suffix)
    if sibling != output:
        fig.savefig(sibling, dpi=200)
        written.append(sibling)
    return written


def plot_pair_split(
    visibility: dict,
    eta_ch12: dict,
    eta_ch13: dict,
    output: Path,
    median_floor: float = DEFAULT_MEDIAN_FLOOR,
) -> list[Path]:
    """Render the two-panel eta_G-insensitivity figure.

    Left panel: the large per-detector-pair Malus-visibility split (a downstream
    coincidence-path contrast parameter). Right panel: the operational QRNG bias
    floor eta_emp under the two coincidence-pair selections, which is invariant
    to ~1e-4 despite the ~0.5 visibility spread -- the empirical fingerprint of
    the eta_G-free nested-loop bound (Theorem nested / Remark asymmetry).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax_v, ax_e) = plt.subplots(1, 2, figsize=(9.2, 3.9), constrained_layout=True)

    # Panel A: downstream detector quality (Malus fringe visibility) per pair.
    pairs = ["Ch1&3", "Ch2&3", "Ch1&4", "Ch2&4"]
    means = [visibility["stagewise_visibility_mean"][pair] for pair in pairs]
    stds = [visibility["stagewise_visibility_std"][pair] for pair in pairs]
    colors = ["#b6412c", "#b6412c", "#2a4d8f", "#2a4d8f"]
    xs = list(range(len(pairs)))
    ax_v.bar(
        xs,
        means,
        yerr=stds,
        capsize=4,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.9,
    )
    for x, mean in zip(xs, means):
        ax_v.text(x, mean + 0.03, f"{mean:.2f}", ha="center", va="bottom", fontsize=9)
    delta_v = abs(
        visibility["grouped_visibility_mean"]["detector4_pairs"]
        - visibility["grouped_visibility_mean"]["detector3_pairs"]
    )
    ax_v.set_xticks(xs)
    ax_v.set_xticklabels(pairs)
    ax_v.set_ylim(0.0, 1.05)
    ax_v.set_ylabel(r"Malus fringe visibility $V$")
    ax_v.set_title(
        "Downstream detector quality\n"
        rf"(varies by $\Delta V \approx {delta_v:.2f}$)",
        fontsize=10,
    )

    # Panel B: operational bias floor eta_emp under the two pair selections.
    labels = [eta_ch12["label"], eta_ch13["label"]]
    etas = [eta_ch12["eta_emp"], eta_ch13["eta_emp"]]
    xe = list(range(len(labels)))
    ax_e.bar(
        xe,
        etas,
        width=0.6,
        color=["#3f7f5f", "#7a4d8c"],
        edgecolor="black",
        linewidth=0.6,
        alpha=0.9,
    )
    for x, eta in zip(xe, etas):
        ax_e.text(
            x,
            eta + median_floor * 0.03,
            rf"${eta * 1e3:.2f}\times10^{{-3}}$",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax_e.axhline(
        median_floor,
        color="#222222",
        linestyle="--",
        linewidth=1.0,
        label=rf"median $\eta_{{\mathrm{{floor}}}}={median_floor * 1e3:.0f}\times10^{{-3}}$",
    )
    delta_eta = abs(etas[0] - etas[1])
    ax_e.set_xticks(xe)
    ax_e.set_xticklabels(labels)
    ax_e.set_ylim(0.0, median_floor * 1.25)
    ax_e.set_ylabel(r"QRNG bias floor $\eta_{\mathrm{emp}}$")
    ax_e.set_title(
        "Operational bias floor\n"
        rf"(invariant to $|\Delta\eta_{{\mathrm{{emp}}}}|\approx{delta_eta * 1e4:.1f}\times10^{{-4}}$)",
        fontsize=10,
    )
    ax_e.legend(loc="upper right", fontsize=8, framealpha=0.9)

    written = _save_figure(fig, output)
    plt.close(fig)
    return written


def main() -> int:
    args = parse_args()

    scans = _load_scans(args.bbo_dir, args.bbo_glob)
    visibility = _summarize_visibility(scans, args.representative_index)
    eta_ch12 = _load_eta_summary(args.stattest_dir, "QEKG Ch1+2", DEFAULT_CH12)
    eta_ch13 = _load_eta_summary(args.stattest_dir, "QEKG Ch1+3", DEFAULT_CH13)

    _print_summary(visibility, eta_ch12, eta_ch13)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "visibility": visibility,
            "eta_emp": {
                "ch1_plus_2": eta_ch12,
                "ch1_plus_3": eta_ch13,
                "absolute_difference": abs(eta_ch12["eta_emp"] - eta_ch13["eta_emp"]),
            },
        }
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"\nWrote {args.json_out}")

    if not args.no_plot:
        try:
            written = plot_pair_split(
                visibility, eta_ch12, eta_ch13, args.plot_out, args.median_floor
            )
        except ImportError as exc:
            print(f"\nSkipped figure ({exc}); install matplotlib or pass --no-plot.")
        else:
            print("\nWrote " + ", ".join(str(path) for path in written))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())