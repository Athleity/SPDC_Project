#!/usr/bin/env python
"""
Sparsity-based covariance denoising for ECMBI tomography CSVs.

Input CSV schema (per file):
  basis, TT, d1, d9, t_meas, t_spent

This script:
1) Loads all CSV files in a directory
2) Builds a measurement matrix from TT (rows=files, cols=bases)
3) Computes sample covariance
4) Denoises with an ℓ1-regularized estimator (Graphical Lasso)
5) Reports an SNR-style improvement metric
6) Computes an EPR entanglement criterion (configurable basis→quadrature mapping)
7) Saves before/after covariance plots
"""

from __future__ import annotations

import argparse
import json
import math
import os
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
    from sklearn.covariance import GraphicalLasso
except Exception:  # pragma: no cover
    GraphicalLasso = None  # type: ignore[assignment]

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
class EprMapping:
    """Mapping from basis names to quadratures (x1, p1, x2, p2)."""

    x1: str
    p1: str
    x2: str
    p2: str


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


def sample_covariance(X: np.ndarray, ddof: int = 1) -> np.ndarray:
    # Rows: samples (files), Cols: variables (bases)
    # Center columns before covariance.
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / max(1, (X.shape[0] - ddof))
    cov = 0.5 * (cov + cov.T)
    return cov


def graphical_lasso_denoise(X: np.ndarray, alpha: float, max_iter: int) -> np.ndarray:
    if GraphicalLasso is None:
        _require("scikit-learn")
    # sklearn expects samples x features.
    model = GraphicalLasso(alpha=alpha, max_iter=max_iter)
    model.fit(X)
    cov = np.asarray(model.covariance_, dtype=float)
    cov = 0.5 * (cov + cov.T)
    return cov


def snr_like_metric(cov_signal: np.ndarray, cov_reference: np.ndarray) -> float:
    """
    A simple, reproducible SNR-like metric for covariance denoising.

    Define:
      signal_power = ||cov_signal||_F^2
      noise_power  = ||cov_reference - cov_signal||_F^2
      SNR_dB = 10*log10(signal_power/noise_power)

    With cov_signal = denoised covariance, cov_reference = sample covariance.
    """
    signal_power = float(np.linalg.norm(cov_signal, ord="fro") ** 2)
    noise_power = float(np.linalg.norm(cov_reference - cov_signal, ord="fro") ** 2)
    eps = 1e-12
    return 10.0 * math.log10((signal_power + eps) / (noise_power + eps))


def find_epr_mapping(
    basis_names: List[str],
    mapping_json: Optional[Path],
    explicit: Optional[EprMapping],
) -> Optional[EprMapping]:
    if explicit is not None:
        return explicit

    if mapping_json is not None:
        data = json.loads(mapping_json.read_text(encoding="utf-8"))
        try:
            m = EprMapping(
                x1=str(data["x1"]),
                p1=str(data["p1"]),
                x2=str(data["x2"]),
                p2=str(data["p2"]),
            )
        except KeyError as e:
            raise SystemExit(
                f"{mapping_json}: missing key {e}. Expected keys: x1, p1, x2, p2."
            )
        return m

    # Heuristic inference: look for basis labels that contain these tokens.
    # This is intentionally conservative; if it fails, the script will still run without EPR.
    def pick(token: str) -> Optional[str]:
        candidates = [b for b in basis_names if token.lower() in b.lower()]
        return candidates[0] if len(candidates) == 1 else None

    x1 = pick("x1") or pick("X1")
    p1 = pick("p1") or pick("P1")
    x2 = pick("x2") or pick("X2")
    p2 = pick("p2") or pick("P2")
    if all([x1, p1, x2, p2]):
        return EprMapping(x1=x1, p1=p1, x2=x2, p2=p2)
    return None


def epr_reid_product(cov: np.ndarray, basis_names: List[str], m: EprMapping) -> Dict[str, float]:
    """
    Reid EPR criterion using conditional variances:
      V_inf(x1|x2) = V(x1) - Cov(x1,x2)^2 / V(x2)
      V_inf(p1|p2) = V(p1) - Cov(p1,p2)^2 / V(p2)
      EPR = V_inf(x1|x2) * V_inf(p1|p2)

    For appropriately normalized quadratures (vacuum variance = 1/2), entanglement is often
    indicated by EPR < 1/4. For vacuum variance = 1, threshold is EPR < 1.

    This function returns both thresholds to interpret later.
    """
    idx = {b: i for i, b in enumerate(basis_names)}
    for key in (m.x1, m.p1, m.x2, m.p2):
        if key not in idx:
            raise SystemExit(f"EPR mapping basis '{key}' not found in basis list.")

    i_x1, i_p1, i_x2, i_p2 = idx[m.x1], idx[m.p1], idx[m.x2], idx[m.p2]

    def vinf(i: int, j: int) -> float:
        v_i = float(cov[i, i])
        v_j = float(cov[j, j])
        c_ij = float(cov[i, j])
        if v_j <= 0:
            return float("nan")
        return v_i - (c_ij * c_ij) / v_j

    v_x = vinf(i_x1, i_x2)
    v_p = vinf(i_p1, i_p2)
    epr = v_x * v_p
    return {
        "Vinf_x1_given_x2": v_x,
        "Vinf_p1_given_p2": v_p,
        "EPR_product": epr,
        "threshold_if_vacuum_var_1": 1.0,
        "threshold_if_vacuum_var_half": 0.25,
    }


def save_cov_heatmap(
    cov: np.ndarray,
    basis_names: List[str],
    out_path: Path,
    title: str,
    vmax_quantile: float = 0.98,
) -> None:
    if plt is None:
        _require("matplotlib")
    if sns is None:
        _require("seaborn")

    v = np.abs(cov)
    vmax = float(np.quantile(v, vmax_quantile)) if v.size else 1.0
    vmax = max(vmax, 1e-12)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    sns.heatmap(
        cov,
        ax=ax,
        cmap="vlag",
        center=0.0,
        vmin=-vmax,
        vmax=vmax,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "covariance"},
    )
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_cov_diag_plot(sample_cov: np.ndarray, denoised_cov: np.ndarray, out_path: Path) -> None:
    if plt is None:
        _require("matplotlib")

    diag_s = np.diag(sample_cov)
    diag_d = np.diag(denoised_cov)

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(diag_s, label="sample diag", linewidth=1.5)
    ax.plot(diag_d, label="denoised diag", linewidth=1.5)
    ax.set_xlabel("basis index")
    ax.set_ylabel("variance (cov diag)")
    ax.set_title("Covariance diagonal (variances)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_cov_eigs_plot(sample_cov: np.ndarray, denoised_cov: np.ndarray, out_path: Path) -> None:
    if plt is None:
        _require("matplotlib")

    es = np.sort(np.linalg.eigvalsh(sample_cov))[::-1]
    ed = np.sort(np.linalg.eigvalsh(denoised_cov))[::-1]

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.semilogy(np.maximum(es, 1e-18), label="sample eigvals", linewidth=1.5)
    ax.semilogy(np.maximum(ed, 1e-18), label="denoised eigvals", linewidth=1.5)
    ax.set_xlabel("eigenvalue rank")
    ax.set_ylabel("eigenvalue (log scale)")
    ax.set_title("Covariance eigenvalues")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="L1-regularized covariance denoising for TT measurements.")
    p.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the 12 CSV files.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="GraphicalLasso regularization strength (higher => sparser precision).",
    )
    p.add_argument(
        "--max-iter",
        type=int,
        default=200,
        help="Max iterations for GraphicalLasso.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output directory for plots/results (default: <data-dir>/sparsity_denoising_outputs/<timestamp>).",
    )
    p.add_argument(
        "--epr-map-json",
        type=str,
        default="",
        help="Path to JSON file mapping quadratures: {x1,p1,x2,p2} -> basis names.",
    )
    p.add_argument(
        "--assume-vacuum-var",
        type=float,
        default=1.0,
        help="For printing an EPR pass/fail: use 1.0 if vacuum variance=1, or 0.5 if vacuum variance=1/2.",
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

    # 3) sample covariance
    cov_sample = sample_covariance(X, ddof=1)

    # 4) L1 denoising (Graphical Lasso)
    cov_denoised = graphical_lasso_denoise(X, alpha=float(args.alpha), max_iter=int(args.max_iter))

    # 5) SNR-style metric
    snr_db = snr_like_metric(cov_denoised, cov_sample)

    # 6) EPR criterion (optional)
    mapping_json = Path(args.epr_map_json) if args.epr_map_json else None
    epr_map = find_epr_mapping(basis_names, mapping_json=mapping_json, explicit=None)
    epr_result: Optional[Dict[str, float]] = None
    if epr_map is not None:
        epr_result = epr_reid_product(cov_denoised, basis_names, epr_map)

    # 7) plots
    save_cov_heatmap(
        cov_sample,
        basis_names,
        out_dir / "cov_sample_heatmap.png",
        title=f"Sample covariance (TT) | files={len(file_names)} bases={len(basis_names)}",
    )
    save_cov_heatmap(
        cov_denoised,
        basis_names,
        out_dir / "cov_denoised_heatmap.png",
        title=f"Denoised covariance (GraphicalLasso α={args.alpha:g})",
    )
    save_cov_heatmap(
        cov_sample - cov_denoised,
        basis_names,
        out_dir / "cov_residual_heatmap.png",
        title="Residual (sample - denoised)",
    )
    save_cov_diag_plot(cov_sample, cov_denoised, out_dir / "cov_diagonal.png")
    save_cov_eigs_plot(cov_sample, cov_denoised, out_dir / "cov_eigenvalues.png")

    # Save artifacts
    pd.DataFrame(X, index=file_names, columns=basis_names).to_csv(out_dir / "measurement_matrix_TT.csv")
    pd.DataFrame(cov_sample, index=basis_names, columns=basis_names).to_csv(out_dir / "cov_sample.csv")
    pd.DataFrame(cov_denoised, index=basis_names, columns=basis_names).to_csv(out_dir / "cov_denoised.csv")

    summary = {
        "data_dir": str(data_dir),
        "n_files": int(X.shape[0]),
        "n_bases": int(X.shape[1]),
        "graphical_lasso_alpha": float(args.alpha),
        "snr_like_db": float(snr_db),
        "epr_mapping": None if epr_map is None else {"x1": epr_map.x1, "p1": epr_map.p1, "x2": epr_map.x2, "p2": epr_map.p2},
        "epr_result": epr_result,
        "assume_vacuum_var": float(args.assume_vacuum_var),
        "epr_pass_fail": None,
        "output_dir": str(out_dir),
        "outputs": [
            "cov_sample_heatmap.png",
            "cov_denoised_heatmap.png",
            "cov_residual_heatmap.png",
            "cov_diagonal.png",
            "cov_eigenvalues.png",
            "measurement_matrix_TT.csv",
            "cov_sample.csv",
            "cov_denoised.csv",
            "summary.json",
        ],
    }

    if epr_result is not None:
        thr = 1.0 if float(args.assume_vacuum_var) == 1.0 else 0.25
        summary["epr_pass_fail"] = bool(epr_result["EPR_product"] < thr)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Done.")
    print(f"- Output dir: {out_dir}")
    print(f"- SNR-like (dB): {snr_db:.2f}")
    if epr_result is None:
        print("- EPR: not computed (provide --epr-map-json or ensure bases include unique x1,p1,x2,p2 tokens).")
    else:
        epr = epr_result["EPR_product"]
        print(f"- EPR product (denoised): {epr:.6g}")
        if summary["epr_pass_fail"] is not None:
            print(f"- EPR criterion met (assume vacuum var={args.assume_vacuum_var:g}): {summary['epr_pass_fail']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

