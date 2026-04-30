"""
Sparsity-Driven Entanglement Detection – Full Pipeline
=======================================================
Implements the method from:
  "Sparsity-Driven Entanglement Detection in High-Dimensional Quantum States"
  arXiv:2511.12546

DATA FORMAT (your ECMBI / EMCCD files)
---------------------------------------
Each CSV has a header row and exactly 36 data rows, one per measurement basis:

  Unnamed: 0, basis,  d1,     d9,     TT,     t_meas, t_spent
  0,          h,h,    472880, 543739, 1421,   5,      5.134049
  0,          h,v,    471909, 522575, 223381, 5,      5.168696
  ...

Important columns:
  'basis'  – two-photon measurement basis string  (e.g. 'h,h', 'h,v', ...)
  'TT'     – two-fold coincidence counts          (the signal we analyse)

Dataset layout  (12 files total):
  6 pump powers × 2 Bell states = 12 files

Measurement matrix X (shape: 12 files × 36 bases)
  • Each ROW    = one file  (one experimental run)
  • Each COLUMN = one basis measurement (ordered by the 36 basis labels)

Covariance matrix Σ = cov(X.T)  → shape (36 × 36)
  • Element Σ[i,j] = covariance between basis i and basis j across all files
  • The SPDC correlation peak sits on the main diagonal (same-basis pairs
    covary strongly) and near-diagonal (similar basis pairs).

SNR Definition  (arXiv:2511.12546 §III + Supplementary)
---------------------------------------------------------
  SNR = μ_peak / σ_background

  Peak region      : |i − j| ≤ PEAK_HALF_WIDTH   (near-diagonal of Σ)
  Background region: |i − j| >  BG_START          (far off-diagonal of Σ)

  μ_peak       = mean  of |Σ[i,j]| in the peak region
  σ_background = stdev of |Σ[i,j]| in the background region

  Lasso zeroes out background entries → σ_background drops → SNR rises.

Usage
-----
  python sparsity_denoising.py --data_dir PATH_TO_CSVS --output_dir OUTPUT_PATH

  # Optional tuning flags:
  python sparsity_denoising.py \\
      --data_dir   "D:/SPDC_Project/03_Data/ECMBI_Tomography" \\
      --output_dir "D:/SPDC_Project/03_Data/ECMBI_Tomography/paper_results" \\
      --peak_half_width 3 \\
      --bg_start 6

File naming convention for pump-power parsing:
  ECMBI_<power>mW_<label>.csv      e.g.  ECMBI_50mW_phi_plus.csv
  or any name containing  <digits>mW   e.g.  run_100mW_a.csv
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive backend – safe on all platforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ──────────────────────────────────────────────────────────────────────────────
# 0.  Global configuration  (overridable via CLI)
# ──────────────────────────────────────────────────────────────────────────────

N_BASES         = 36    # expected number of basis rows per CSV
TT_COLUMN       = "TT"  # column that holds two-fold coincidence counts

PEAK_HALF_WIDTH = 3     # |i−j| ≤ this  →  "peak region" in Σ
BG_START        = 6     # |i−j| >  this  →  "background region" in Σ

LASSO_CV_FOLDS  = 5
LASSO_ALPHAS    = np.logspace(-4, 1, 60)   # regularisation search grid
RANDOM_STATE    = 42


# ──────────────────────────────────────────────────────────────────────────────
# 1.  File I/O
# ──────────────────────────────────────────────────────────────────────────────

def parse_pump_power(filepath: str) -> float:
    """
    Extract pump power in mW from the filename.

    Tries, in order:
      1. Pattern  <digits[.digits]>mW   e.g. '50mW', '0.5mW', '100mW'
      2. Pattern  _<digits>_            e.g. '_50_' as fallback
      3. Returns 0.0 with a warning if neither matches.
    """
    stem  = Path(filepath).stem
    match = re.search(r"(\d+(?:\.\d+)?)\s*mW", stem, re.IGNORECASE)
    if match:
        return float(match.group(1))
    match = re.search(r"_(\d+)_", stem)
    if match:
        return float(match.group(1))
    print(f"  [WARN] Cannot parse pump power from '{stem}', assigning 0.0 mW")
    return 0.0


def load_csv_file(filepath: str) -> tuple[np.ndarray, list[str]]:
    """
    Load one ECMBI CSV file and return the TT vector and ordered basis labels.

    Expected file structure
    -----------------------
    Header row  : Unnamed: 0, basis, d1, d9, TT, t_meas, t_spent
    Data rows   : 36 rows, one per two-photon measurement basis

    Returns
    -------
    tt_vector : np.ndarray, shape (36,)
        Two-fold coincidence counts, ordered by the 'basis' column.
    bases : list[str], length 36
        Basis labels in file order (used to align columns across files).
    """
    df = pd.read_csv(filepath, header=0)          # ← correct: use header row

    # ── Validate required columns ────────────────────────────────────────────
    missing = [c for c in ("basis", TT_COLUMN) if c not in df.columns]
    if missing:
        raise ValueError(
            f"File '{Path(filepath).name}' is missing required column(s): {missing}\n"
            f"  Found columns: {list(df.columns)}"
        )

    # ── Validate row count ───────────────────────────────────────────────────
    if len(df) != N_BASES:
        print(
            f"  [WARN] '{Path(filepath).name}' has {len(df)} rows "
            f"(expected {N_BASES}). Using all rows."
        )

    # ── Sort by basis label for consistent column ordering across files ──────
    df = df.sort_values("basis").reset_index(drop=True)

    bases     = df["basis"].tolist()
    tt_vector = df[TT_COLUMN].values.astype(float)

    return tt_vector, bases


def load_dataset(data_dir: str) -> dict[float, dict]:
    """
    Scan data_dir for CSV files, load each one, and group by pump power.

    Returns
    -------
    dataset : dict  {pump_power_mW: {"X": np.ndarray, "bases": list[str]}}

        X has shape (n_files_for_that_power, n_bases)
          • Each ROW    = one experimental file / run
          • Each COLUMN = one basis setting

        bases : canonical ordered list of basis labels (36 strings)
    """
    pattern = os.path.join(data_dir, "*.csv")
    files   = sorted(glob.glob(pattern, recursive=False))

    if not files:
        raise FileNotFoundError(
            f"\nNo CSV files found under '{data_dir}'.\n"
            "Please check the path or use --data_dir to point at your data folder."
        )

    # First pass: load all files, collect TT vectors per pump power
    groups: dict[float, list[np.ndarray]] = {}
    canonical_bases: list[str] | None = None

    for fp in files:
        power             = parse_pump_power(fp)
        tt_vector, bases  = load_csv_file(fp)

        # Use the first file's basis order as the canonical reference
        if canonical_bases is None:
            canonical_bases = bases
        elif bases != canonical_bases:
            print(
                f"  [WARN] '{Path(fp).name}' has different basis order – "
                "reordering to match canonical order."
            )
            # Re-align: build a mapping from label → TT value, then reorder
            basis_to_tt = dict(zip(bases, tt_vector))
            tt_vector   = np.array(
                [basis_to_tt.get(b, np.nan) for b in canonical_bases]
            )

        groups.setdefault(power, []).append(tt_vector)
        print(
            f"  Loaded  {Path(fp).name:<40s}  "
            f"power={power:>6.1f} mW   TT range=[{tt_vector.min():.0f}, {tt_vector.max():.0f}]"
        )

    # Stack rows for each pump power
    dataset: dict[float, dict] = {}
    for power, vectors in groups.items():
        X = np.vstack(vectors)           # shape: (n_files, n_bases)
        dataset[power] = {"X": X, "bases": canonical_bases}
        print(
            f"\n  → Pump {power:>6.1f} mW : "
            f"{X.shape[0]} files × {X.shape[1]} bases  "
            f"(X shape {X.shape})"
        )

    return dataset


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Covariance
# ──────────────────────────────────────────────────────────────────────────────

def compute_covariance(X: np.ndarray) -> np.ndarray:
    """
    Sample covariance matrix  Σ = cov(X.T),  shape (n_bases × n_bases).

    X  has shape  (n_files, n_bases).
    np.cov(X.T) treats each column of X.T (= each basis) as a variable,
    giving Σ[i,j] = covariance of basis i and basis j across all files.

    With only 12 files (rows) the sample covariance is noisy – that is
    exactly what the ℓ1 regularisation corrects.
    """
    return np.cov(X.T)   # shape: (n_bases, n_bases)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  ℓ1-regularised sparse reconstruction
# ──────────────────────────────────────────────────────────────────────────────

def lasso_denoise(Sigma: np.ndarray) -> np.ndarray:
    """
    Sparse reconstruction of the covariance matrix via column-wise LassoCV.

    Following arXiv:2511.12546:
      For each column j, fit:
          Σ[:, j]  ≈  Σ[:, k≠j] · β_j   (ℓ1-penalised)
      The ℓ1 penalty drives small / noise-driven coefficients to zero,
      keeping only the physically meaningful correlations.

    Steps
    -----
    1. Standardise Σ column-wise (zero mean, unit variance) so the Lasso
       alpha grid is scale-independent.
    2. For each column j, regress it against all other columns with LassoCV.
    3. Reconstruct the full column from the Lasso fit.
    4. Undo standardisation to recover original units.
    5. Symmetrise: Σ_hat = (Σ_hat + Σ_hat.T) / 2.
    6. Hard-threshold near-zero off-diagonal entries (Lasso may leave
       tiny residuals instead of exact zeros).

    Returns
    -------
    Sigma_denoised : np.ndarray, same shape as Sigma, symmetric.
    """
    n      = Sigma.shape[0]
    scaler = StandardScaler(with_mean=True, with_std=True)

    # Work in normalised space so alpha search is scale-free
    Sigma_norm     = scaler.fit_transform(Sigma)   # (n, n)
    Sigma_denoised = np.zeros_like(Sigma_norm)

    for j in range(n):
        y      = Sigma_norm[:, j]                          # target column
        X_feat = np.delete(Sigma_norm, j, axis=1)          # all other columns

        lasso = LassoCV(
            alphas        = LASSO_ALPHAS,
            cv            = min(LASSO_CV_FOLDS, len(y)),   # can't exceed n_samples
            fit_intercept = True,
            max_iter      = 10_000,
            random_state  = RANDOM_STATE,
            n_jobs        = -1,
        )
        lasso.fit(X_feat, y)

        # X_feat has n rows; lasso.predict(X_feat) → length n.
        # Do NOT use np.insert — just overwrite position j with original y[j].
        y_pred               = lasso.predict(X_feat)   # shape (n,)
        y_pred[j]            = y[j]                    # restore self-covariance
        Sigma_denoised[:, j] = y_pred

    # Undo StandardScaler:  x_orig = x_norm * std + mean
    Sigma_denoised = (
        Sigma_denoised * scaler.scale_[np.newaxis, :]
        + scaler.mean_[np.newaxis, :]
    )

    # Symmetrise
    Sigma_denoised = (Sigma_denoised + Sigma_denoised.T) / 2.0

    # Hard-threshold tiny residual off-diagonal entries to exact zeros
    # Threshold = 1% of mean diagonal magnitude (keeps variance, kills noise)
    diag_mean = np.abs(np.diag(Sigma_denoised)).mean()
    threshold  = 0.01 * diag_mean if diag_mean > 0 else 0.0
    if threshold > 0:
        off_diag_small = (np.abs(Sigma_denoised) < threshold)
        np.fill_diagonal(off_diag_small, False)   # never zero the diagonal
        Sigma_denoised[off_diag_small] = 0.0

    return Sigma_denoised


# ──────────────────────────────────────────────────────────────────────────────
# 4.  SNR calculation  (paper definition)
# ──────────────────────────────────────────────────────────────────────────────

def calculate_snr(
    cov_matrix      : np.ndarray,
    peak_half_width : int = PEAK_HALF_WIDTH,
    bg_start        : int = BG_START,
) -> float:
    """
    SNR = μ_peak / σ_background   (arXiv:2511.12546, §III + Supplementary)

    Physical motivation
    -------------------
    Σ[i, j] is the covariance between basis i and basis j.
    When the bases are ordered consistently (e.g. alphabetically), the
    SPDC correlation peak runs along the main diagonal:
      •  Same basis (i = j)   : maximum covariance (photon pairs detected
         together → counts rise and fall together across pump-power files).
      •  Similar bases nearby : moderate covariance.
      •  Distant bases        : should be near zero (background noise).

    In the "difference coordinate"  Δ = |i − j|:
      Peak region      (Δ ≤ peak_half_width) → μ_peak
      Background region(Δ >  bg_start)       → σ_background

    Parameters
    ----------
    cov_matrix      : square covariance matrix (n × n)
    peak_half_width : half-width of the diagonal peak in index units
    bg_start        : index offset beyond which we call entries "background"

    Returns
    -------
    float : SNR value (higher = better signal visibility)
    """
    n       = cov_matrix.shape[0]
    abs_cov = np.abs(cov_matrix)

    # Matrix of index-difference values
    idx  = np.arange(n)
    diff = np.abs(idx[:, None] - idx[None, :])   # shape (n, n)

    peak_mask = (diff <= peak_half_width)
    bg_mask   = (diff >  bg_start)

    peak_vals = abs_cov[peak_mask]
    bg_vals   = abs_cov[bg_mask]

    mu_peak  = float(peak_vals.mean()) if peak_vals.size > 0 else 0.0
    sigma_bg = float(bg_vals.std())    if bg_vals.size   > 0 else 0.0

    if sigma_bg == 0.0:
        return np.inf if mu_peak > 0.0 else 0.0
    return mu_peak / sigma_bg


def snr_improvement_dB(snr_raw: float, snr_denoised: float) -> float:
    """
    SNR improvement in decibels:  ΔSNRdB = 20 · log10(SNR_denoised / SNR_raw)

    Positive value → denoising improved signal visibility (the expected result).
    """
    if snr_raw <= 0.0 or not np.isfinite(snr_raw):
        return 0.0
    if not np.isfinite(snr_denoised):
        return float("inf")
    return 20.0 * np.log10(snr_denoised / snr_raw)


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Visualisation
# ──────────────────────────────────────────────────────────────────────────────

def plot_heatmaps(
    Sigma_raw     : np.ndarray,
    Sigma_denoised: np.ndarray,
    bases         : list[str],
    power_mW      : float,
    snr_raw       : float,
    snr_den       : float,
    output_dir    : str,
    label         : str = "",
) -> str:
    """
    Side-by-side heatmaps of raw and denoised covariance matrices.
    Tick labels show the basis names for interpretability.
    """
    n    = len(bases)
    vmax = max(np.abs(Sigma_raw).max(), 1e-9)

    # Show at most 12 tick labels (thin matrices are fine to label fully)
    tick_step  = max(1, n // 12)
    tick_idx   = list(range(0, n, tick_step))
    tick_labels= [bases[i] for i in tick_idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, mat, title, snr in zip(
        axes,
        [Sigma_raw, Sigma_denoised],
        [f"Raw Covariance\nSNR = {snr_raw:.3f}",
         f"Denoised (ℓ₁-Lasso)\nSNR = {snr_den:.3f}"],
        [snr_raw, snr_den],
    ):
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       origin="upper", interpolation="nearest", aspect="auto")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Basis j", fontsize=10)
        ax.set_ylabel("Basis i", fontsize=10)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(tick_idx)
        ax.set_yticklabels(tick_labels, fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="Covariance (counts²)")

    delta_dB = snr_improvement_dB(snr_raw, snr_den)
    fig.suptitle(
        f"Pump power: {power_mW:.1f} mW  {label}  │  "
        f"SNR improvement: {delta_dB:+.2f} dB",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()

    safe_label = label.replace(" ", "_").replace("/", "-")
    fname = os.path.join(
        output_dir,
        f"heatmap_{power_mW:.1f}mW{('_' + safe_label) if safe_label else ''}.png",
    )
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


def plot_snr_vs_power(results: list[dict], output_dir: str) -> str:
    """Line + scatter plot of raw and denoised SNR vs pump power."""
    if len(results) < 2:
        return ""

    powers    = sorted(set(r["power_mW"] for r in results))
    snr_raw_  = {}
    snr_den_  = {}

    for r in results:
        p = r["power_mW"]
        snr_raw_.setdefault(p, []).append(r["snr_raw"])
        snr_den_.setdefault(p, []).append(r["snr_denoised"])

    p_arr    = np.array(powers)
    raw_mean = np.array([np.mean(snr_raw_[p]) for p in powers])
    den_mean = np.array([np.mean(snr_den_[p]) for p in powers])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(p_arr, raw_mean, "o--", color="#e74c3c", label="Raw covariance",   linewidth=1.5)
    ax.plot(p_arr, den_mean, "s-",  color="#2980b9", label="After ℓ₁ denoising", linewidth=2)
    ax.set_xlabel("Pump power (mW)", fontsize=11)
    ax.set_ylabel("SNR  (μ_peak / σ_background)", fontsize=11)
    ax.set_title("SNR vs. Pump Power: Raw vs. ℓ₁-Regularised", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()

    fname = os.path.join(output_dir, "snr_vs_power.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


def plot_improvement_bar(results: list[dict], output_dir: str) -> str:
    """Bar chart of SNR improvement (dB) for every pump-power group."""
    if len(results) < 1:
        return ""

    labels  = [f"{r['power_mW']:.0f} mW" for r in results]
    deltas  = [r["improvement_dB"] for r in results]
    colors  = ["#27ae60" if d >= 0 else "#e74c3c" for d in deltas]
    max_abs = max((abs(d) for d in deltas if np.isfinite(d)), default=1.0)

    fig, ax = plt.subplots(figsize=(max(6, len(results) * 1.1), 4))
    bars = ax.bar(labels, deltas, color=colors, edgecolor="k", linewidth=0.6)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Pump power")
    ax.set_ylabel("SNR improvement (dB)")
    ax.set_title("ℓ₁-Regularisation SNR Improvement vs. Pump Power")

    for bar, val in zip(bars, deltas):
        if not np.isfinite(val):
            continue
        ypos = bar.get_height() + 0.04 * max_abs
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{val:+.1f} dB", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fname = os.path.join(output_dir, "snr_improvement_bar.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(data_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    banner = "  Sparsity-Driven Entanglement Denoising Pipeline  "
    print(f"\n{'='*len(banner)}")
    print(banner)
    print(f"{'='*len(banner)}")
    print(f"  Data dir    : {data_dir}")
    print(f"  Output dir  : {output_dir}")
    print(f"  TT column   : '{TT_COLUMN}'")
    print(f"  Peak ½-width: {PEAK_HALF_WIDTH}  basis indices")
    print(f"  BG start    : {BG_START}  basis indices")
    print(f"{'='*len(banner)}\n")

    # ── 6.1  Load all CSV files ───────────────────────────────────────────────
    print("► Loading CSV files …\n")
    dataset = load_dataset(data_dir)
    print(f"\n  Total pump-power groups: {len(dataset)}\n")

    results: list[dict] = []

    for power_mW in sorted(dataset.keys()):
        X     = dataset[power_mW]["X"]
        bases = dataset[power_mW]["bases"]
        n_files, n_bases = X.shape

        sep = "─" * 60
        print(f"\n{sep}")
        print(f"  Pump power : {power_mW:.1f} mW")
        print(f"  X shape    : {X.shape}  ({n_files} files × {n_bases} bases)")
        print(f"  TT range   : [{X.min():.0f}, {X.max():.0f}] counts")

        # ── 6.2  Warn if very few samples ────────────────────────────────────
        if n_files < 4:
            print(
                f"  [WARN] Only {n_files} file(s) for this power level. "
                "Covariance estimate will be unreliable. "
                "Ideally you want ≥ 6 files per power."
            )

        # ── 6.3  Raw covariance ───────────────────────────────────────────────
        print("\n  ► Computing sample covariance Σ₀ …")
        Sigma_raw = compute_covariance(X)          # (n_bases × n_bases)
        snr_raw   = calculate_snr(Sigma_raw)
        print(f"    Σ₀ shape  : {Sigma_raw.shape}")
        print(f"    SNR (raw) : {snr_raw:.4f}")

        # ── 6.4  ℓ1 sparse reconstruction ────────────────────────────────────
        print("\n  ► ℓ₁-regularised sparse reconstruction (LassoCV) …")
        print( "    (This may take 10–60 seconds for a 36×36 matrix)")
        Sigma_den = lasso_denoise(Sigma_raw)
        snr_den   = calculate_snr(Sigma_den)
        delta_dB  = snr_improvement_dB(snr_raw, snr_den)
        print(f"    SNR (denoised) : {snr_den:.4f}")
        print(f"    Improvement    : {delta_dB:+.2f} dB")

        # ── 6.5  Save covariance matrices as CSV ──────────────────────────────
        stem    = f"cov_{power_mW:.1f}mW"
        raw_csv = os.path.join(output_dir, f"{stem}_raw.csv")
        den_csv = os.path.join(output_dir, f"{stem}_denoised.csv")

        # Include basis labels as header/index for readability
        pd.DataFrame(Sigma_raw, index=bases, columns=bases).to_csv(raw_csv)
        pd.DataFrame(Sigma_den, index=bases, columns=bases).to_csv(den_csv)
        print(f"\n  ► Saved: {raw_csv}")
        print(f"  ► Saved: {den_csv}")

        # ── 6.6  Heatmaps ────────────────────────────────────────────────────
        fig_path = plot_heatmaps(
            Sigma_raw, Sigma_den, bases,
            power_mW, snr_raw, snr_den,
            output_dir,
        )
        print(f"  ► Saved: {fig_path}")

        results.append(dict(
            power_mW      = power_mW,
            n_files       = n_files,
            n_bases       = n_bases,
            snr_raw       = snr_raw,
            snr_denoised  = snr_den,
            improvement_dB= delta_dB,
        ))

    # ── 6.7  Summary printout ─────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("  RESULTS SUMMARY")
    print(f"{'='*70}")
    hdr = f"  {'Power (mW)':>10}  {'Files':>5}  {'SNR raw':>10}  {'SNR denoised':>13}  {'Δ SNR':>10}"
    print(hdr)
    print(f"  {'-'*10}  {'-'*5}  {'-'*10}  {'-'*13}  {'-'*10}")
    for r in results:
        print(
            f"  {r['power_mW']:>10.1f}  "
            f"{r['n_files']:>5d}  "
            f"{r['snr_raw']:>10.4f}  "
            f"{r['snr_denoised']:>13.4f}  "
            f"{r['improvement_dB']:>+9.2f} dB"
        )
    print(f"{'='*70}\n")

    # ── 6.8  Summary figures ──────────────────────────────────────────────────
    f1 = plot_snr_vs_power(results, output_dir)
    f2 = plot_improvement_bar(results, output_dir)
    if f1: print(f"► Saved: {f1}")
    if f2: print(f"► Saved: {f2}")

    # ── 6.9  Save results table ───────────────────────────────────────────────
    res_csv = os.path.join(output_dir, "snr_results.csv")
    pd.DataFrame(results).to_csv(res_csv, index=False)
    print(f"► Saved: {res_csv}")

    print("\n✓  Pipeline complete.\n")


# ──────────────────────────────────────────────────────────────────────────────
# 7.  CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Sparsity-driven entanglement denoising (arXiv:2511.12546)\n"
            "Reads ECMBI CSV files, builds TT covariance matrix,\n"
            "applies ℓ1 regularisation, and reports SNR improvement."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data_dir", default="data",
        help="Folder containing the ECMBI CSV files (default: ./data)"
    )
    parser.add_argument(
        "--output_dir", default="output",
        help="Destination folder for CSV and PNG outputs (default: ./output)"
    )
    parser.add_argument(
        "--tt_column", default=TT_COLUMN,
        help=f"Name of the coincidence-count column (default: '{TT_COLUMN}')"
    )
    parser.add_argument(
        "--peak_half_width", type=int, default=PEAK_HALF_WIDTH,
        help=(
            f"Half-width (in basis-index units) of the correlation peak region "
            f"(default: {PEAK_HALF_WIDTH})"
        )
    )
    parser.add_argument(
        "--bg_start", type=int, default=BG_START,
        help=(
            f"Basis-index offset beyond which entries are counted as background "
            f"(default: {BG_START})"
        )
    )
    parser.add_argument(
        "--n_bases", type=int, default=N_BASES,
        help=f"Expected number of basis rows per CSV file (default: {N_BASES})"
    )

    args = parser.parse_args()

    # Push CLI values into globals so all functions see them
    TT_COLUMN       = args.tt_column
    PEAK_HALF_WIDTH = args.peak_half_width
    BG_START        = args.bg_start
    N_BASES         = args.n_bases

    run_pipeline(args.data_dir, args.output_dir)