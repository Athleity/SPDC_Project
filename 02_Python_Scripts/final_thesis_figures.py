"""
Thesis figure generator (real ECMBI tomography CSVs).

Fig2 mode:
  - A: Bell-parameter style curve vs pump power (matches `bell_S_vs_power.py` heuristic)
  - B: Fidelity vs pump power (matches `analyze_all_powers.py`)
  - C: Both A and B in a 1x2 panel

Default is C unless overridden by env var `SPDC_FIG2_MODE` in {a,b,c}.
"""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.linalg import sqrtm
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config

warnings.filterwarnings("ignore", category=UserWarning)

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
plt.rcParams.update(
    {
        "font.size": 12,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    }
)


Fig2Mode = Literal["A", "B", "C"]


def _fig2_mode_from_env() -> Fig2Mode:
    raw = os.environ.get("SPDC_FIG2_MODE", "c").strip().lower()
    if raw in {"a", "b", "c"}:
        return raw.upper()  # type: ignore[return-value]
    return "C"


FIG2_MODE: Fig2Mode = _fig2_mode_from_env()


def _normalize_basis_label(raw: object) -> str:
    """Strip CSV quotes/whitespace so 'h,h', '\"h,h\"', etc. match canonical keys."""
    s = str(raw).strip().strip('"').strip("'")
    return s


def _basis_tt(df: pd.DataFrame, basis: str) -> float:
    want = _normalize_basis_label(basis)
    bnorm = df["basis"].map(_normalize_basis_label)
    rows = df.loc[bnorm == want, "TT"]
    if rows.empty:
        return 0.0
    return float(pd.to_numeric(rows, errors="coerce").fillna(0.0).iloc[0])


def _basis_csv_to_proj_key(basis_norm: str) -> str:
    """Map tomography CSV label 'h,v' -> projector key 'HV' (matches `quantum_tomography.py`)."""
    parts = _normalize_basis_label(basis_norm).split(",")
    if len(parts) != 2:
        return ""
    lut = {"h": "H", "v": "V", "d": "D", "a": "A", "r": "R", "l": "L"}
    a = parts[0].strip().lower()
    b = parts[1].strip().lower()
    if a not in lut or b not in lut:
        return ""
    return lut[a] + lut[b]


def bell_state_phi_plus() -> np.ndarray:
    rho = np.zeros((4, 4), dtype=complex)
    rho[0, 0] = 0.5
    rho[3, 3] = 0.5
    rho[0, 3] = 0.5
    rho[3, 0] = 0.5
    return rho


def bell_state_phi_minus() -> np.ndarray:
    """|Φ−⟩ = (|HH⟩ − |VV⟩)/√2."""
    rho = np.zeros((4, 4), dtype=complex)
    rho[0, 0] = 0.5
    rho[3, 3] = 0.5
    rho[0, 3] = -0.5
    rho[3, 0] = -0.5
    return rho


def bell_state_psi_plus() -> np.ndarray:
    """|Ψ+⟩ = (|HV⟩ + |VH⟩)/√2."""
    rho = np.zeros((4, 4), dtype=complex)
    rho[1, 1] = 0.5
    rho[2, 2] = 0.5
    rho[1, 2] = 0.5
    rho[2, 1] = 0.5
    return rho


def bell_state_psi_minus() -> np.ndarray:
    """|Ψ⁻⟩ = (|HV⟩ − |VH⟩)/√2 in the ordering [HH, HV, VH, VV]."""
    rho = np.zeros((4, 4), dtype=complex)
    rho[1, 1] = 0.5
    rho[2, 2] = 0.5
    rho[1, 2] = -0.5
    rho[2, 1] = -0.5
    return rho


def create_projectors() -> Dict[str, np.ndarray]:
    H = np.array([1, 0], dtype=complex)
    V = np.array([0, 1], dtype=complex)
    D = np.array([1, 1], dtype=complex) / np.sqrt(2)
    A = np.array([1, -1], dtype=complex) / np.sqrt(2)
    R = np.array([1, -1j], dtype=complex) / np.sqrt(2)
    L = np.array([1, 1j], dtype=complex) / np.sqrt(2)

    states = {"H": H, "V": V, "D": D, "A": A, "R": R, "L": L}

    projectors: Dict[str, np.ndarray] = {}
    for name1, vec1 in states.items():
        proj1 = np.outer(vec1, vec1.conj())
        for name2, vec2 in states.items():
            proj2 = np.outer(vec2, vec2.conj())
            key = f"{name1}{name2}"
            projectors[key] = np.kron(proj1, proj2)
    return projectors


_PROJECTORS_CACHE: Optional[Dict[str, np.ndarray]] = None


def _projectors() -> Dict[str, np.ndarray]:
    global _PROJECTORS_CACHE
    if _PROJECTORS_CACHE is None:
        _PROJECTORS_CACHE = create_projectors()
    return _PROJECTORS_CACHE


def reconstruct_density_matrix(measurements: Dict[str, float], projectors: Dict[str, np.ndarray]) -> np.ndarray:
    total = float(sum(max(0.0, float(v)) for v in measurements.values()))
    if total <= 0:
        return np.eye(4, dtype=complex) / 4.0

    exp_probs = {k: max(0.0, float(measurements.get(k, 0.0))) / total for k in projectors.keys()}

    def rho_from_T(T: np.ndarray) -> np.ndarray:
        T_mat = T.reshape(4, 4)
        rho = T_mat.conj().T @ T_mat
        tr = float(np.real(np.trace(rho)))
        if tr <= 0:
            return np.eye(4, dtype=complex) / 4.0
        return rho / tr

    def neg_log_likelihood(T: np.ndarray) -> float:
        rho = rho_from_T(T)
        log_lik = 0.0
        for key, proj in projectors.items():
            pred = float(np.real(np.trace(rho @ proj)))
            pred = max(1e-10, min(1.0, pred))
            if exp_probs[key] > 0:
                log_lik += exp_probs[key] * np.log(pred)
        return float(-log_lik)

    T0 = (np.eye(4).flatten() * 0.5).astype(float)
    result = minimize(neg_log_likelihood, T0, method="L-BFGS-B", options={"maxiter": 1200})
    rho = rho_from_T(result.x)
    rho = (rho + rho.conj().T) / 2
    rho = rho / np.trace(rho)
    return rho


def calculate_fidelity(rho: np.ndarray, target: np.ndarray) -> float:
    sqrt_target = sqrtm(target)
    sqrt_target_rho = sqrt_target @ rho @ sqrt_target
    evals = np.linalg.eigvals(sqrt_target_rho)
    return float(np.real((np.sum(np.sqrt(np.maximum(0.0, np.real(evals))))) ** 2))


def tomographic_bell_fidelity(df: pd.DataFrame, _state: str) -> float:
    """
    Quantum fidelity from the full 36-setting coincidence vector.

    We report the **best overlap over the four Bell states** {|Φ±⟩, |Ψ±⟩} (basis-ordering matches
    `quantum_tomography.py`).
    """
    projectors = _projectors()
    measurements: Dict[str, float] = {k: 0.0 for k in projectors.keys()}
    for _, row in df.iterrows():
        key = _basis_csv_to_proj_key(str(row["basis"]))
        if not key:
            continue
        measurements[key] = measurements.get(key, 0.0) + float(pd.to_numeric(row["TT"], errors="coerce") or 0.0)

    try:
        rho = reconstruct_density_matrix(measurements, projectors)
    except Exception:
        return float("nan")

    try:
        bell_targets = (
            bell_state_phi_plus(),
            bell_state_phi_minus(),
            bell_state_psi_plus(),
            bell_state_psi_minus(),
        )
        return float(max(np.real(calculate_fidelity(rho, t)) for t in bell_targets))
    except Exception:
        return float("nan")


def _tt_column_to_basis_label(tt_col: str) -> str:
    """Data column 'TT__h,h' -> plot label 'h,h'."""
    if tt_col.startswith("TT__"):
        return tt_col[len("TT__") :]
    return tt_col


def _sklearn_feature_to_basis_label(name: str) -> Optional[str]:
    """
    sklearn output like 'num__TT__h,h' -> 'h,h'.
    Handles odd tokens defensively via regex.
    """
    s = str(name)
    m = re.match(r"^num__TT__(.+)$", s)
    if m:
        return m.group(1)
    return None


def _parse_power_mw(filename: str) -> int:
    m = re.search(r"(\d+)mW", filename, flags=re.IGNORECASE)
    return int(m.group(1)) if m else 0


def _parse_state(filename: str) -> str:
    u = filename.upper()
    if "PSIPLUS" in u:
        return "PsiPlus"
    if "PSIMINUS" in u:
        return "PsiMinus"
    return "Unknown"


def _load_ecmbi_tomography_frames(data_dir: Path) -> List[Tuple[str, pd.DataFrame]]:
    frames: List[Tuple[str, pd.DataFrame]] = []
    for p in sorted(data_dir.glob("*.csv")):
        # Skip obvious non-tomography exports if they appear at the folder root.
        if p.name.lower().startswith("measurement_matrix"):
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if "basis" not in df.columns or "TT" not in df.columns:
            continue
        frames.append((p.name, df))
    return frames


def _per_file_hhvv_table(files: List[Tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for fname, df in files:
        state = _parse_state(fname)
        hh = _basis_tt(df, "h,h")
        hv = _basis_tt(df, "h,v")
        vh = _basis_tt(df, "v,h")
        vv = _basis_tt(df, "v,v")
        total = hh + hv + vh + vv
        # Raw diagonal fraction in the four H/V coincidence channels (often ≪ 1 when HV/VH dominate).
        hv_fraction_diagonal = (hh + vv) / total if total > 0 else 0.0
        e_hv = (hh + vv - hv - vh) / total if total > 0 else 0.0
        s_est = 2.828 * abs(e_hv)
        f_tomo = tomographic_bell_fidelity(df, state) if state != "Unknown" else float("nan")

        rows.append(
            {
                "filename": fname,
                "power_mW": _parse_power_mw(fname),
                "state": state,
                "HH": hh,
                "HV": hv,
                "VH": vh,
                "VV": vv,
                "total_HV_basis": total,
                "hv_diagonal_fraction": hv_fraction_diagonal,
                "fidelity_Bell_tomography": f_tomo,
                "E_hv": e_hv,
                "S_estimate": s_est,
            }
        )
    return pd.DataFrame(rows).sort_values(["power_mW", "state", "filename"]).reset_index(drop=True)


def _wide_feature_row(fname: str, df: pd.DataFrame) -> Dict[str, object]:
    # Pivot all 36 bases into columns for ML feature construction.
    dfp = df[["basis", "TT"]].copy()
    dfp["basis"] = dfp["basis"].astype(str)
    dfp["TT"] = pd.to_numeric(dfp["TT"], errors="coerce").fillna(0.0)
    piv = dfp.set_index("basis")["TT"]
    out: Dict[str, object] = {f"TT__{k}": float(v) for k, v in piv.items()}
    out["power_mW"] = float(_parse_power_mw(fname))
    out["state"] = _parse_state(fname)
    return out


def _build_modeling_table(files: List[Tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for fname, df in files:
        row = _wide_feature_row(fname, df)
        row["filename"] = fname
        rows.append(row)
    wide = pd.DataFrame(rows)
    # Ensure stable column order: filename + meta + sorted TT features
    tt_cols = sorted([c for c in wide.columns if c.startswith("TT__")])
    meta = ["filename", "power_mW", "state"]
    return wide[meta + tt_cols]


def _make_preprocessor(feature_cols: List[str]) -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, feature_cols),
            ("cat", categorical, ["state"]),
        ]
    )


def _leave_one_pump_power_out_predict(model_tbl: pd.DataFrame, model) -> np.ndarray:
    """
    Train without any measurements at pump power P, predict rows where power == P.

    This evaluates extrapolation across pump powers (much harder than LOO on rows), and yields
    RMSE/MAE scales comparable to tens of mW for sparse power grids.
    """
    feature_cols = [c for c in model_tbl.columns if c.startswith("TT__")]
    cols = feature_cols + ["state"]
    y = model_tbl["power_mW"].astype(float).values
    pred = np.zeros_like(y, dtype=float)

    pw = model_tbl["power_mW"].astype(float).values
    powers = sorted(set(pw.tolist()))
    for p in powers:
        train_idx = np.flatnonzero(pw != p)
        test_idx = np.flatnonzero(pw == p)
        if train_idx.size == 0 or test_idx.size == 0:
            continue

        X_train = model_tbl.iloc[train_idx][cols]
        y_train = model_tbl.iloc[train_idx]["power_mW"].astype(float).values
        X_test = model_tbl.iloc[test_idx][cols]

        pipe = Pipeline(
            steps=[
                ("prep", _make_preprocessor(feature_cols)),
                ("model", model),
            ]
        )
        pipe.fit(X_train, y_train)
        pred[test_idx] = pipe.predict(X_test)

    return pred


def _lop_o_predictions_rf(model_tbl: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_cols = [c for c in model_tbl.columns if c.startswith("TT__")]
    if not feature_cols:
        raise ValueError("No TT__ feature columns found.")
    y = model_tbl["power_mW"].astype(float).values
    y_hat = _leave_one_pump_power_out_predict(
        model_tbl,
        RandomForestRegressor(n_estimators=900, random_state=42),
    )
    return y, y_hat, y - y_hat


def _lop_o_metrics_table(model_tbl: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in model_tbl.columns if c.startswith("TT__")]
    cols = feature_cols + ["state"]
    y = model_tbl["power_mW"].astype(float).values
    pw = model_tbl["power_mW"].astype(float).values

    models = {
        "Random Forest": RandomForestRegressor(n_estimators=900, random_state=42),
        "Neural Network": MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=8000, random_state=42),
        "Linear Regression": LinearRegression(),
    }

    rows = []
    for name, est in models.items():
        pred = np.zeros_like(y, dtype=float)
        ok = True
        powers = sorted(set(pw.tolist()))
        try:
            for p in powers:
                train_idx = np.flatnonzero(pw != p)
                test_idx = np.flatnonzero(pw == p)
                if train_idx.size == 0 or test_idx.size == 0:
                    continue
                X_train = model_tbl.iloc[train_idx][cols]
                y_train = model_tbl.iloc[train_idx]["power_mW"].astype(float).values
                X_test = model_tbl.iloc[test_idx][cols]
                pipe = Pipeline(steps=[("prep", _make_preprocessor(feature_cols)), ("model", est)])
                pipe.fit(X_train, y_train)
                pred[test_idx] = pipe.predict(X_test)
        except Exception:
            ok = False

        if ok:
            rows.append(
                {
                    "model": name,
                    "R2": float(r2_score(y, pred)),
                    "RMSE_mW": float(np.sqrt(mean_squared_error(y, pred))),
                    "MAE_mW": float(mean_absolute_error(y, pred)),
                }
            )
        else:
            rows.append({"model": name, "R2": float("nan"), "RMSE_mW": float("nan"), "MAE_mW": float("nan")})
    return pd.DataFrame(rows)


def _rf_feature_importance(model_tbl: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in model_tbl.columns if c.startswith("TT__")]
    X = model_tbl[feature_cols + ["state"]]
    y = model_tbl["power_mW"].astype(float).values

    pipe = Pipeline(
        steps=[
            ("prep", _make_preprocessor(feature_cols)),
            ("model", RandomForestRegressor(n_estimators=1200, random_state=42)),
        ]
    )
    pipe.fit(X, y)

    model: RandomForestRegressor = pipe.named_steps["model"]  # type: ignore[assignment]
    prep: ColumnTransformer = pipe.named_steps["prep"]  # type: ignore[assignment]

    feat_names = list(prep.get_feature_names_out())
    imp = model.feature_importances_
    out = pd.DataFrame({"feature": feat_names, "importance": imp}).sort_values("importance", ascending=False)

    collapsed = []
    for _, r in out.iterrows():
        name = str(r["feature"])
        if name.startswith("cat__onehot__"):
            collapsed.append({"feature": f"state ({name.split('cat__onehot__', 1)[1]})", "importance": float(r["importance"])})
            continue
        basis_name = _sklearn_feature_to_basis_label(name)
        if basis_name is not None:
            collapsed.append({"feature": basis_name, "importance": float(r["importance"])})
        else:
            collapsed.append({"feature": name, "importance": float(r["importance"])})

    cdf = pd.DataFrame(collapsed)
    cdf = cdf.groupby("feature", as_index=False)["importance"].sum().sort_values("importance", ascending=False)
    cdf["importance_pct"] = 100.0 * cdf["importance"] / cdf["importance"].sum() if cdf["importance"].sum() > 0 else 0.0
    return cdf


def _plot_fig1_power_prediction(y: np.ndarray, y_hat: np.ndarray, out_dir: Path) -> None:
    r2 = float(r2_score(y, y_hat))
    rmse = float(np.sqrt(mean_squared_error(y, y_hat)))

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(y, y_hat, s=120, c="royalblue", edgecolors="black", linewidth=1.0, alpha=0.85, zorder=3)

    lo = float(min(y.min(), y_hat.min()))
    hi = float(max(y.max(), y_hat.max()))
    pad = max(5.0, 0.05 * (hi - lo))
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "r--", linewidth=2.2, label="Ideal (y = x)", alpha=0.75, zorder=1)

    ax.text(
        0.05,
        0.95,
        f"R² = {r2:.3f}\nRMSE = {rmse:.1f} mW\n(leave-one-pump-power-out)",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85),
    )

    ax.set_xlabel("Actual pump power (mW)", fontweight="bold")
    ax.set_ylabel("Predicted pump power (mW)", fontweight="bold")
    ax.set_title("Pump-power extrapolation from tomography coincidences (Random Forest)", fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(out_dir / "Fig1_Power_Prediction.png", bbox_inches="tight")
    fig.savefig(out_dir / "Fig1_Power_Prediction.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_fig2_bell_and_or_fidelity(summary: pd.DataFrame, out_dir: Path) -> None:
    avg = (
        summary.groupby("power_mW", as_index=False)
        .agg({"S_estimate": "mean", "fidelity_Bell_tomography": "mean"})
        .sort_values("power_mW")
    )

    fig = plt.figure(figsize=(14, 6))

    if FIG2_MODE == "A":
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(avg["power_mW"], avg["S_estimate"], "ro-", linewidth=2.5, markersize=8, label="S estimate (2√2·|E|)")
        ax.axhline(2.0, color="b", linestyle="--", linewidth=2.0, label="Classical reference (2)")
        ax.axhline(2.828, color="g", linestyle=":", linewidth=2.0, label="Tsirelson bound (2√2)")
        ax.set_xlabel("Pump power (mW)", fontweight="bold")
        ax.set_ylabel("S estimate", fontweight="bold")
        ax.set_title("Fig 2A: Bell-inequality-style metric vs pump power (HV-basis correlation proxy)", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    elif FIG2_MODE == "B":
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(
            avg["power_mW"],
            avg["fidelity_Bell_tomography"],
            "bo-",
            linewidth=2.5,
            markersize=8,
            label="max F(ρ, Bell₄) (36-setting reconstruction)",
        )
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("Pump power (mW)", fontweight="bold")
        ax.set_ylabel("Bell-state fidelity", fontweight="bold")
        ax.set_title("Fig 2B: Bell-state fidelity vs pump power (max over four Bell states)", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    else:  # C
        axl = fig.add_subplot(1, 2, 1)
        axl.plot(avg["power_mW"], avg["S_estimate"], "ro-", linewidth=2.5, markersize=8)
        axl.axhline(2.0, color="b", linestyle="--", linewidth=2.0)
        axl.axhline(2.828, color="g", linestyle=":", linewidth=2.0)
        axl.set_xlabel("Pump power (mW)", fontweight="bold")
        axl.set_ylabel("S estimate", fontweight="bold")
        axl.set_title("Fig 2C (left): S estimate vs pump power", fontweight="bold")
        axl.grid(True, alpha=0.3)

        axr = fig.add_subplot(1, 2, 2)
        axr.plot(avg["power_mW"], avg["fidelity_Bell_tomography"], "bo-", linewidth=2.5, markersize=8)
        axr.set_ylim(0.0, 1.05)
        axr.set_xlabel("Pump power (mW)", fontweight="bold")
        axr.set_ylabel("Bell-state fidelity", fontweight="bold")
        axr.set_title("Fig 2C (right): Bell fidelity vs pump power", fontweight="bold")
        axr.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "Fig2_Bell_Parameter.png", bbox_inches="tight")
    fig.savefig(out_dir / "Fig2_Bell_Parameter.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_fig3_model_performance(metrics: pd.DataFrame, out_dir: Path) -> None:
    m = metrics.dropna(subset=["R2", "RMSE_mW", "MAE_mW"], how="all").copy()
    if m.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

    axes[0].bar(m["model"], m["R2"], color="steelblue", alpha=0.85, edgecolor="black")
    axes[0].set_ylabel("R² (leave-one pump power)", fontweight="bold")
    axes[0].set_title("Model accuracy", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(m["model"], m["RMSE_mW"], color="coral", alpha=0.85, edgecolor="black")
    axes[1].set_ylabel("RMSE (mW)", fontweight="bold")
    axes[1].set_title("RMSE when each pump power is held out", fontweight="bold")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(True, alpha=0.3, axis="y")

    axes[2].bar(m["model"], m["MAE_mW"], color="lightgreen", alpha=0.85, edgecolor="black")
    axes[2].set_ylabel("MAE (mW)", fontweight="bold")
    axes[2].set_title("MAE when each pump power is held out", fontweight="bold")
    axes[2].tick_params(axis="x", rotation=25)
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        "Pump-power extrapolation models (leave-one pump power held out per fold)",
        fontweight="bold",
        fontsize=14,
    )
    plt.tight_layout()
    fig.savefig(out_dir / "Fig3_Model_Performance.png", bbox_inches="tight")
    fig.savefig(out_dir / "Fig3_Model_Performance.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_fig4_correlation(model_tbl: pd.DataFrame, out_dir: Path) -> None:
    # Use a compact set: a few strong bases + label meta (not included in corr plot).
    tt_cols = [c for c in model_tbl.columns if c.startswith("TT__")]
    # Pick the 8 bases with highest variance across files (more informative than arbitrary picks).
    variances = model_tbl[tt_cols].var(axis=0).sort_values(ascending=False)
    pick = list(variances.head(8).index)
    if len(pick) < 2:
        return

    sub = model_tbl[pick].copy()
    sub.columns = [_tt_column_to_basis_label(c) for c in sub.columns]

    corr = sub.corr(method="pearson", min_periods=1)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Pearson correlation", fontweight="bold")

    ticks = np.arange(len(corr.columns))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(list(corr.columns), rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(list(corr.columns), fontsize=10)

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            val = float(corr.values[i, j])
            text_color = "white" if abs(val) > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=9)

    ax.set_title("Correlation matrix (basis coincidence channels)", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "Fig4_Correlation_Matrix.png", bbox_inches="tight")
    fig.savefig(out_dir / "Fig4_Correlation_Matrix.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_fig5_feature_importance(importances: pd.DataFrame, out_dir: Path) -> None:
    top = importances.head(12).copy()
    top = top.sort_values("importance_pct", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["feature"], top["importance_pct"], color="steelblue", alpha=0.85, edgecolor="black")
    ax.set_xlabel("Importance (%)", fontweight="bold")
    ax.set_ylabel("Feature", fontweight="bold")
    ax.set_title("Random Forest feature importance (pump-power model)", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    fig.savefig(out_dir / "Fig5_Feature_Importance.png", bbox_inches="tight")
    fig.savefig(out_dir / "Fig5_Feature_Importance.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_fig6_residuals(y: np.ndarray, resid: np.ndarray, out_dir: Path) -> None:
    y_hat = y - resid

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 5))

    axa.scatter(y_hat, resid, s=90, c="purple", alpha=0.75, edgecolors="black")
    axa.axhline(0.0, color="red", linestyle="--", linewidth=2)
    axa.set_xlabel("Predicted pump power (mW)", fontweight="bold")
    axa.set_ylabel("Residuals (mW)", fontweight="bold")
    axa.set_title("Residual plot (leave-one pump power)", fontweight="bold")
    axa.grid(True, alpha=0.3)

    axb.hist(resid, bins=min(12, max(5, len(np.unique(resid)))), color="skyblue", alpha=0.75, edgecolor="black")
    axb.axvline(0.0, color="red", linestyle="--", linewidth=2)
    axb.set_xlabel("Residuals (mW)", fontweight="bold")
    axb.set_ylabel("Count", fontweight="bold")
    axb.set_title("Residual distribution", fontweight="bold")
    axb.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Model validation: residual analysis", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "Fig6_Residual_Analysis.png", bbox_inches="tight")
    fig.savefig(out_dir / "Fig6_Residual_Analysis.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    config.ensure_dirs(config.THESIS_FIG_DIR, config.RESULTS_CSV_DIR)

    data_dir = config.ECMBI_TOMOGRAPHY_DIR
    files = _load_ecmbi_tomography_frames(data_dir)
    if not files:
        raise FileNotFoundError(
            f"No usable ECMBI tomography CSVs found in: {data_dir}\n"
            f"Expected root files like PSI*twoQBtomo_*mW_*.csv with columns basis,TT."
        )

    summary = _per_file_hhvv_table(files)
    model_tbl = _build_modeling_table(files).reset_index(drop=True)

    y, y_hat, resid = _lop_o_predictions_rf(model_tbl)
    metrics = _lop_o_metrics_table(model_tbl)
    importances = _rf_feature_importance(model_tbl)

    # Save CSV outputs
    summary.to_csv(config.RESULTS_CSV_DIR / "thesis_ecmbi_per_file_summary.csv", index=False)
    (
        summary.groupby("power_mW", as_index=False)
        .agg({"S_estimate": "mean", "fidelity_Bell_tomography": "mean", "E_hv": "mean"})
        .sort_values("power_mW")
        .to_csv(config.RESULTS_CSV_DIR / "thesis_ecmbi_averaged_by_power.csv", index=False)
    )
    model_tbl.to_csv(config.RESULTS_CSV_DIR / "thesis_ecmbi_modeling_table.csv", index=False)
    metrics.to_csv(config.RESULTS_CSV_DIR / "thesis_model_cv_metrics.csv", index=False)
    importances.to_csv(config.RESULTS_CSV_DIR / "thesis_rf_feature_importances.csv", index=False)
    pd.DataFrame({"actual_power_mW": y, "predicted_power_mW_oof": y_hat, "residual_mW": resid}).to_csv(
        config.RESULTS_CSV_DIR / "thesis_power_prediction_oof.csv", index=False
    )

    # Figures
    _plot_fig1_power_prediction(y, y_hat, config.THESIS_FIG_DIR)
    _plot_fig2_bell_and_or_fidelity(summary, config.THESIS_FIG_DIR)
    _plot_fig3_model_performance(metrics, config.THESIS_FIG_DIR)
    _plot_fig4_correlation(model_tbl, config.THESIS_FIG_DIR)
    _plot_fig5_feature_importance(importances, config.THESIS_FIG_DIR)
    _plot_fig6_residuals(y, resid, config.THESIS_FIG_DIR)

    print("=" * 72)
    print("THESIS FIGURES GENERATED (REAL ECMBI DATA)")
    print(f"Fig2 mode: {FIG2_MODE} (set SPDC_FIG2_MODE=a|b|c to change)")
    print(f"Figures: {config.THESIS_FIG_DIR}")
    print(f"CSVs:    {config.RESULTS_CSV_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
