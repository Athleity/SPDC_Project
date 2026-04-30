#!/usr/bin/env python
"""
Correlation denoising for ECMBI tomography CSVs (small-sample friendly).

Input CSV schema (per file):
  basis, TT, d1, d9, t_meas, t_spent

Your dataset is typically 12 files (samples) × 36 bases (features), so the
sample covariance is singular. This script avoids SPD solvers and instead
denoises the *correlation matrix* via sparse Lasso neighborhood regression.

What it does:
1) Load all CSVs in a directory
2) Build measurement matrix from TT (rows=files, cols=bases)
3) Group files by pump power parsed from filename (configurable heuristic)
4) For each pump power group:
   - compute sample correlation
   - denoise correlation using Lasso regressions (works when n << p)
   - compute SNR before/after using a bootstrap reference correlation
   - save correlation heatmaps (before/after/residual)
5) Save a summary bar chart of SNR improvement across pump powers
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def _require(pkg: str) -> None:
    raise SystemExit(
        f"Missing dependency '{pkg}'. Install it (e.g. `pip install {pkg}`) and re-run."
    )


try:
    from sklearn.linear_model import Lasso
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover
    Lasso = None  # type: ignore[assignment]
    StandardScaler = None  # type: ignore[assignment]

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None  # type: ignore[assignment]

try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None  # type: ignore[assignment]


DEFAULT_DATA_DIR = r"D:\SPDC_Project\03_Data\ECMBI_Tomography"


@dataclass(frozen=True)
class PumpGroup:
    key: str
    files: List[str]


def _parse_pump_power_key(filename: str) -> str:
    """
    Try to extract a pump power token from the filename.

    Examples it can parse:
      - "..._120mW_..." -> "120mW"
      - "p=80mw.csv"    -> "80mW"
      - "45.5mW_run2"   -> "45.5mW"

    If nothing matches, it falls back to the filename stem.
    """
    stem = Path(filename).stem
    m = re.search(r"(\d+(?:\.\d+)?)\s*(mW|mw|uW|uw|nW|nw)", stem)
    if not m:
        return stem
    value = m.group(1)
    unit = m.group(2)
    unit_norm = unit[0] + "W"  # mW/uW/nW
    return f"{value}{unit_norm}"


def list_csvs(data_dir: Path) -> List[Path]:
    csvs = sorted([p for p in data_dir.glob("*.csv") if p.is_file()])
    if not csvs:
        raise SystemExit(f"No CSV files found in: {data_dir}")
    return csvs


def load_one_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"basis", "TT"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{path.name}: missing required columns: {sorted(missing)}")
    if df["basis"].isna().any():
        raise SystemExit(f"{path.name}: 'basis' contains NaNs")
    if df["TT"].isna().any():
        raise SystemExit(f"{path.name}: 'TT' contains NaNs")
    return df


def load_all_csvs(csv_paths: Iterable[Path]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for p in csv_paths:
        out[p.name] = load_one_csv(p)
    return out


def infer_basis_order(dfs: Dict[str, pd.DataFrame]) -> List[str]:
    # Use the first file as the canonical ordering (36 measurement bases expected).
    first = next(iter(dfs.values()))
    bases = first["basis"].astype(str).tolist()
    # Ensure consistent basis sets across files.
    base_set = set(bases)
    for name, df in dfs.items():
        b = df["basis"].astype(str).tolist()
        if set(b) != base_set:
            raise SystemExit(
                "Basis sets differ across files.\n"
                f"First file bases count={len(base_set)}; {name} bases count={len(set(b))}."
            )
    return bases


def build_measurement_matrix(
    dfs: Dict[str, pd.DataFrame], basis_order: List[str]
) -> Tuple[np.ndarray, List[str], List[str]]:
    file_names = sorted(dfs.keys())
    basis_index = {b: i for i, b in enumerate(basis_order)}
    X = np.full((len(file_names), len(basis_order)), np.nan, dtype=float)

    for r, fname in enumerate(file_names):
        df = dfs[fname]
        for _, row in df.iterrows():
            b = str(row["basis"])
            c = basis_index.get(b)
            if c is None:
                continue
            X[r, c] = float(row["TT"])

    if np.isnan(X).any():
        nan_locs = np.argwhere(np.isnan(X))
        example = nan_locs[0]
        raise SystemExit(
            "Measurement matrix contains NaNs (missing TT for some basis/file).\n"
            f"Example NaN at row={example[0]} (file={file_names[example[0]]}), "
            f"col={example[1]} (basis={basis_order[example[1]]})."
        )

    return X, file_names, basis_order


def sample_correlation(X: np.ndarray) -> np.ndarray:
    """
    Sample Pearson correlation of features (columns) across samples (rows).
    Safe for n < p (correlation is defined even if covariance is singular).
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    s = Xc.std(axis=0, ddof=1)
    s = np.where(s <= 0, 1.0, s)
    Z = Xc / s
    R = (Z.T @ Z) / max(1, (Z.shape[0] - 1))
    R = np.clip(0.5 * (R + R.T), -1.0, 1.0)
    np.fill_diagonal(R, 1.0)
    return R


def _standardize(X: np.ndarray) -> np.ndarray:
    if StandardScaler is None:
        _require("scikit-learn")
    scaler = StandardScaler(with_mean=True, with_std=True)
    return scaler.fit_transform(X)


def lasso_denoised_correlation(
    X: np.ndarray,
    alpha: float,
    max_iter: int,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Denoise correlations using Lasso neighborhood regression.

    For each feature j:
      x_j ≈ sum_{k!=j} beta_{j,k} x_k, estimated with Lasso

    We then build a symmetric adjacency-like correlation estimate:
      Rhat_{j,k} = sign(beta_{j,k}) * min(|r_{j,k}|, |beta_{j,k}|)  (heuristic shrink)

    This avoids any SPD requirement and behaves well when n << p.
    """
    if Lasso is None:
        _require("scikit-learn")

    n, p = X.shape
    if n < 3:
        # Too small for stable regression; return sample correlation.
        return sample_correlation(X)

    Z = _standardize(X)
    R = sample_correlation(X)

    B = np.zeros((p, p), dtype=float)
    for j in range(p):
        y = Z[:, j]
        Xj = np.delete(Z, j, axis=1)
        model = Lasso(alpha=alpha, max_iter=max_iter, fit_intercept=False)
        model.fit(Xj, y)
        coef = model.coef_.astype(float, copy=False)
        # Map back into full coefficient vector
        full = np.zeros(p, dtype=float)
        full[np.arange(p) != j] = coef
        B[j, :] = full

    # Symmetric shrinkage using both directions
    Rhat = np.zeros((p, p), dtype=float)
    for j in range(p):
        for k in range(j + 1, p):
            b_jk = B[j, k]
            b_kj = B[k, j]
            raw = R[j, k]
            mag = max(abs(b_jk), abs(b_kj))
            shrunk = math.copysign(min(abs(raw), mag), raw) if mag > eps else 0.0
            Rhat[j, k] = shrunk
            Rhat[k, j] = shrunk
    np.fill_diagonal(Rhat, 1.0)
    Rhat = np.clip(Rhat, -1.0, 1.0)
    return Rhat


def bootstrap_reference_correlation(
    X: np.ndarray,
    n_boot: int,
    seed: int,
) -> np.ndarray:
    """
    Bootstrap reference correlation: average correlation across bootstrap resamples.
    Used to define an SNR-like before/after metric without ground-truth.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n < 3 or n_boot <= 1:
        return sample_correlation(X)
    acc = np.zeros((X.shape[1], X.shape[1]), dtype=float)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n, endpoint=False)
        acc += sample_correlation(X[idx, :])
    Rref = acc / float(n_boot)
    np.fill_diagonal(Rref, 1.0)
    return np.clip(0.5 * (Rref + Rref.T), -1.0, 1.0)


def snr_db_against_reference(R_est: np.ndarray, R_ref: np.ndarray) -> float:
    """
    SNR-like dB versus a reference correlation matrix.

      signal_power = ||R_ref||_F^2
      error_power  = ||R_est - R_ref||_F^2
      SNR_dB = 10*log10(signal_power/error_power)
    """
    signal_power = float(np.linalg.norm(R_ref, ord="fro") ** 2)
    error_power = float(np.linalg.norm(R_est - R_ref, ord="fro") ** 2)
    eps = 1e-12
    return 10.0 * math.log10((signal_power + eps) / (error_power + eps))


def save_corr_heatmap(
    corr: np.ndarray,
    basis_names: List[str],
    out_path: Path,
    title: str,
    vmax_quantile: float = 0.98,
) -> None:
    if plt is None:
        _require("matplotlib")
    if sns is None:
        _require("seaborn")

    v = np.abs(corr)
    vmax = float(np.quantile(v, vmax_quantile)) if v.size else 1.0
    vmax = max(vmax, 1e-12)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    sns.heatmap(
        corr,
        ax=ax,
        cmap="vlag",
        center=0.0,
        vmin=-vmax,
        vmax=vmax,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "correlation"},
    )
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_snr_bar_chart(rows: List[Dict[str, float]], out_path: Path, title: str) -> None:
    if plt is None:
        _require("matplotlib")
    if not rows:
        return
    labels = [str(r["pump_key"]) for r in rows]
    before = [float(r["snr_before_db"]) for r in rows]
    after = [float(r["snr_after_db"]) for r in rows]
    improvement = [a - b for a, b in zip(after, before)]

    x = np.arange(len(labels))
    width = 0.35
    fig = plt.figure(figsize=(max(8, 0.7 * len(labels)), 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(x - width / 2, before, width, label="SNR before (dB)")
    ax.bar(x + width / 2, after, width, label="SNR after (dB)")
    ax.plot(x, improvement, color="black", linewidth=1.5, marker="o", label="Improvement (dB)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("dB")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Small-sample correlation denoising for TT measurements (Lasso-based).")
    p.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the 12 CSV files.",
    )
    p.add_argument(
        "--lasso-alpha",
        type=float,
        default=0.05,
        help="Lasso regularization strength (higher => more sparsity).",
    )
    p.add_argument(
        "--max-iter",
        type=int,
        default=5000,
        help="Max iterations for Lasso solver.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output directory for plots/results (default: <data-dir>/sparsity_denoising_outputs/<timestamp>).",
    )
    p.add_argument(
        "--bootstrap",
        type=int,
        default=300,
        help="Bootstrap resamples used to form a reference correlation for SNR.",
    )
    p.add_argument(
        "--seed",
        type=float,
        default=0,
        help="Random seed for bootstrap.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else data_dir / "sparsity_denoising_outputs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    csvs = list_csvs(data_dir)
    dfs = load_all_csvs(csvs)
    basis_order = infer_basis_order(dfs)
    X, file_names, basis_names = build_measurement_matrix(dfs, basis_order)

    # Group by pump power key parsed from filename
    pump_to_rows: Dict[str, List[int]] = {}
    for i, fname in enumerate(file_names):
        key = _parse_pump_power_key(fname)
        pump_to_rows.setdefault(key, []).append(i)

    # Always save measurement matrix
    pd.DataFrame(X, index=file_names, columns=basis_names).to_csv(out_dir / "measurement_matrix_TT.csv")

    snr_rows: List[Dict[str, float]] = []
    for pump_key in sorted(pump_to_rows.keys()):
        idx = pump_to_rows[pump_key]
        Xg = X[idx, :]
        group_dir = out_dir / f"pump_{pump_key}"
        group_dir.mkdir(parents=True, exist_ok=True)

        R_sample = sample_correlation(Xg)
        R_denoised = lasso_denoised_correlation(
            Xg, alpha=float(getattr(args, "lasso_alpha")), max_iter=int(args.max_iter)
        )
        R_ref = bootstrap_reference_correlation(
            Xg, n_boot=int(args.bootstrap), seed=int(float(args.seed))
        )

        snr_before = snr_db_against_reference(R_sample, R_ref)
        snr_after = snr_db_against_reference(R_denoised, R_ref)
        snr_rows.append(
            {
                "pump_key": pump_key,
                "n_files": float(len(idx)),
                "snr_before_db": float(snr_before),
                "snr_after_db": float(snr_after),
                "snr_improvement_db": float(snr_after - snr_before),
            }
        )

        save_corr_heatmap(
            R_sample,
            basis_names,
            group_dir / "corr_sample_heatmap.png",
            title=f"Sample correlation (TT) | pump={pump_key} | n={len(idx)}",
            vmax_quantile=0.98,
        )
        save_corr_heatmap(
            R_denoised,
            basis_names,
            group_dir / "corr_denoised_heatmap.png",
            title=f"Denoised correlation (Lasso alpha={getattr(args, 'lasso_alpha'):g}) | pump={pump_key}",
            vmax_quantile=0.98,
        )
        save_corr_heatmap(
            R_sample - R_denoised,
            basis_names,
            group_dir / "corr_residual_heatmap.png",
            title=f"Residual (sample - denoised) | pump={pump_key}",
            vmax_quantile=0.98,
        )

        pd.DataFrame(R_sample, index=basis_names, columns=basis_names).to_csv(group_dir / "corr_sample.csv")
        pd.DataFrame(R_denoised, index=basis_names, columns=basis_names).to_csv(group_dir / "corr_denoised.csv")
        pd.DataFrame(R_ref, index=basis_names, columns=basis_names).to_csv(group_dir / "corr_reference_bootstrap_mean.csv")

        (group_dir / "snr.json").write_text(
            json.dumps(
                {
                    "pump_key": pump_key,
                    "n_files": len(idx),
                    "files": [file_names[i] for i in idx],
                    "lasso_alpha": float(getattr(args, "lasso_alpha")),
                    "bootstrap": int(args.bootstrap),
                    "seed": int(float(args.seed)),
                    "snr_before_db": float(snr_before),
                    "snr_after_db": float(snr_after),
                    "snr_improvement_db": float(snr_after - snr_before),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    # Summary artifacts
    snr_df = pd.DataFrame(snr_rows).sort_values("pump_key")
    snr_df.to_csv(out_dir / "snr_summary.csv", index=False)
    save_snr_bar_chart(
        snr_rows,
        out_dir / "snr_summary_bars.png",
        title="SNR before/after denoising by pump power",
    )

    summary = {
        "data_dir": str(data_dir),
        "n_files_total": int(X.shape[0]),
        "n_bases": int(X.shape[1]),
        "groups": {k: [file_names[i] for i in v] for k, v in pump_to_rows.items()},
        "lasso_alpha": float(getattr(args, "lasso_alpha")),
        "bootstrap": int(args.bootstrap),
        "seed": int(float(args.seed)),
        "output_dir": str(out_dir),
        "outputs": [
            "measurement_matrix_TT.csv",
            "snr_summary.csv",
            "snr_summary_bars.png",
            "summary.json",
            "pump_<key>/corr_sample_heatmap.png",
            "pump_<key>/corr_denoised_heatmap.png",
            "pump_<key>/corr_residual_heatmap.png",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Done.")
    print(f"- Output dir: {out_dir}")
    print(f"- Pump groups: {len(pump_to_rows)}")
    if snr_rows:
        best = max(snr_rows, key=lambda r: r["snr_improvement_db"])
        print(f"- Best improvement: pump={best['pump_key']}  dSNR={best['snr_improvement_db']:.2f} dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

