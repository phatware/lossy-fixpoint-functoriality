# Lossy Fixpoint Functoriality

This repository contains the published paper PDF and the reproducible experiment-analysis scripts/data needed to regenerate the figures shipped in the paper.


## Paper overview

[Lossy Fixpoint Functoriality](https://doi.org/10.13140/RG.2.2.25717.74724) develops a 2-categorical tolerance calculus for reasoning about loops that are not preserved exactly. The main idea is to treat loop-preservation error as explicit 2-cell data with a measurable defect, so that approximation bounds can be transported and composed in a disciplined way rather than hidden inside ad hoc estimates. This produces round-trip and nested-loop transport bounds, including an important asymmetric distinction: some constructions pay both forward and return defects, while others are insensitive to one leg of the round trip.

The paper then specializes this abstract framework to entangled-photon protocols, especially the QRNG calibration setting represented by the archived data in this repository. The experimental appendix is not presented as a full validation of the theory, but as a concrete operating-point case where the categorical distinction leaves a visible empirical fingerprint.

## Motivation

This project originated in earlier [Quantum-Classical Recursive Consciousness (QCRC)](https://doi.org/10.13140/RG.2.2.28396.63360) work. In that setting I needed a proper notion of distance for lossy loops in an RC-dagger category: something strong enough to measure deviation from exact loop preservation, but structured enough to compose cleanly. The loop-preservation 2-cell used in this paper is the mathematical device that came out of that need.

The published paper does not foreground the QCRC origin, because I wanted it to stand on its own as a self-contained mathematical paper rather than depend on that earlier context. At the same time, the framework turned out to be useful for revisiting my older experimental archives, giving a principled language for the QRNG calibration, Malus-fringe, LCC-voltage, and related archived analyses collected here.

## Requirements

- Python 3.10+
- packages from `requirements.txt`

Setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Repository layout

```bash
LossyFixpointFunctoriality.pdf
experiments/
  Images/                            # generated figures
  QEKG/TestData/                     # archived QEKG CSV inputs and notebooks
  experiment_data/                   # earlier entropy-test archives
  plot_qrng_calibration.py
  plot_randomness_tests.py
  plot_lcc_calibration.py
  plot_bbo_malus.py
  recompute_pair_visibility_eta.py
```

## Running the scripts

Run from the repository root:

```bash
python experiments/plot_qrng_calibration.py
python experiments/plot_randomness_tests.py --seed 0
python experiments/plot_lcc_calibration.py
python experiments/plot_bbo_malus.py
python experiments/recompute_pair_visibility_eta.py
```

The scripts write figures into `experiments/Images/` by default.

Their default input/output locations are resolved from the script file location, not from the shell's current working directory. That means they work both from the repo root (`python experiments/...`) and from inside `experiments/` (`python plot_qrng_calibration.py`).

## What each script does

- `plot_qrng_calibration.py`: rebuilds the QRNG calibration figure from archived entropy-test CSVs.
- `plot_randomness_tests.py`: regenerates the entropy and monobit comparison panels for QRNG vs a software CSPRNG baseline.
- `plot_lcc_calibration.py`: plots coincidence rates and the MSB asymmetry curve versus LCC voltage.
- `plot_bbo_malus.py`: fits Malus-law coincidence fringes and summarizes stagewise visibility.
- `recompute_pair_visibility_eta.py`: recomputes the visibility split and the `eta_emp` split discussed in the appendix, and regenerates the two-panel comparison figure.

## Citation

If you use this research, please cite:

```bibtex
@article{miasnikov2025rc,
  title={LOSSY FIXPOINT FUNCTORIALITY: A 2-CATEGORICAL TOLERANCE CALCULUS FOR ENTANGLED-PHOTON PROTOCOLS},
  author={Miasnikov, Stanislav},
  year={2026},
  doi={https://doi.org/10.13140/RG.2.2.25717.74724}
}
```
