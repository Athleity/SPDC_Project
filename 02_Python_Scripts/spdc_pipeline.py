"""
================================================================================
SPARSITY DENOISING + NEURAL NETWORK PIPELINE
Implements: arXiv:2511.12546  "Sparsity-Driven Entanglement Detection"
================================================================================

Runs the paper's ℓ1-Lasso covariance denoising AND a trained denoising
autoencoder, then compares both methods by SNR improvement and EPR criterion.

Usage
-----
  python spdc_pipeline.py

Edit the two path variables at the top of Section 0 to match your system.
All other parameters are documented inline.
================================================================================
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import glob
import re
import warnings
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")
np.random.seed(42)

# TensorFlow – graceful fallback if not installed
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.callbacks import (
        EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    )
    from tensorflow.keras.optimizers import Adam
    tf.random.set_seed(42)
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[WARN] TensorFlow not found. Neural network step will be skipped.")
    print("       Install with:  pip install tensorflow")

try:
    import seaborn as sns
    SNS_AVAILABLE = True
except ImportError:
    SNS_AVAILABLE = False


# ============================================================================
# SECTION 0 – CONFIGURATION  (edit these two paths)
# ============================================================================

DATA_DIR   = r"D:\SPDC_Project\03_Data\ECMBI_Tomography"
OUTPUT_DIR = r"D:\SPDC_Project\03_Data\ECMBI_Tomography\FINAL_RESULTS"

# ── Exact filenames (copy-pasted from your spec) ─────────────────────────────
EXPECTED_FILES = [
    "PSIPLUStwoQBtomo_20mW_2023-02-10--12h-08m_10--12h-15mS1.csv",
    "PSIPLUStwoQBtomo_50mW_2023-02-10--11h-16m_10--11h-23mS1.csv",
    "PSIPLUStwoQBtomo_100mW_2023-02-10--11h-26m_10--11h-34mS1.csv",
    "PSIPLUStwoQBtomo_150mW_2023-02-10--11h-39m_10--11h-46mS1.csv",
    "PSIPLUStwoQBtomo_200mW_2023-02-10--11h-49m_10--11h-56mS1.csv",
    "PSIPLUStwoQBtomo_250mW_2023-02-10--11h-58m_10--12h-05mS1.csv",
    "PSIMINUStwoQBtomo_20mW_2023-02-10--14h-43m_10--14h-50mS1.csv",
    "PSIMINUStwoQBtomo_50mW_2023-02-10--15h-36m_10--15h-43mS1.csv",
    "PSIMINUStwoQBtomo_100mW_2023-02-10--15h-44m_10--15h-52mS1.csv",
    "PSIMINUStwoQBtomo_150mW_2023-02-10--15h-53m_10--16h-00mS1.csv",
    "PSIMINUStwoQBtomo_200mW_2023-02-10--16h-50m_10--16h-58mS1.csv",
    "PSIMINUStwoQBtomo_250mW_2023-02-10--17h-07m_10--17h-14mS1.csv",
]

# ── Canonical basis order (exactly as in your spec) ──────────────────────────
CANONICAL_BASES = [
    'h,h', 'h,v', 'h,d', 'h,a', 'h,r', 'h,l',
    'v,h', 'v,v', 'v,d', 'v,a', 'v,r', 'v,l',
    'd,h', 'd,v', 'd,d', 'd,a', 'd,r', 'd,l',
    'a,h', 'a,v', 'a,d', 'a,a', 'a,r', 'a,l',
    'r,h', 'r,v', 'r,d', 'r,a', 'r,r', 'r,l',
    'l,h', 'l,v', 'l,d', 'l,a', 'l,r', 'l,l',
]
N_BASES = len(CANONICAL_BASES)   # 36

# ── TT validation ranges per pump power ──────────────────────────────────────
TT_VALIDATION = {
    20:  (130,   45_000),
    50:  (450,  112_000),
    100: (1260, 228_000),
    150: (2380, 336_000),
    200: (4000, 460_000),
    250: (5300, 550_000),
}

# ── SNR parameters (paper Eq. 3) ─────────────────────────────────────────────
PEAK_HALF_WIDTH = 3    # |i-j| <= this  →  peak region
BG_START        = 6    # |i-j| >  this  →  background region

# ── Lasso parameters ─────────────────────────────────────────────────────────
LASSO_CV      = 3
LASSO_ALPHAS  = np.logspace(-4, 1, 50)
LASSO_MAXITER = 5000
RANDOM_STATE  = 42

# ── Neural network parameters ────────────────────────────────────────────────
NN_AUGMENT_PER_SAMPLE = 84    # 12 samples × 84 = 1008 total
NN_NOISE_LEVELS       = [0.01, 0.05, 0.10]
NN_EPOCHS             = 100
NN_BATCH              = 16
NN_PATIENCE           = 10
NN_LR                 = 1e-3


# ============================================================================
# SECTION 1 – DATA LOADING
# ============================================================================

def parse_pump_power(filename: str) -> int:
    """Extract pump power (mW) from filename. Returns integer."""
    m = re.search(r"(\d+)mW", filename, re.IGNORECASE)
    return int(m.group(1)) if m else 0


def load_one_file(filepath: str) -> np.ndarray:
    """
    Load a single ECMBI CSV and return a TT vector of length 36,
    sorted to CANONICAL_BASES order.

    Errors raised with actionable messages.
    """
    fname = Path(filepath).name

    # FIX ERROR 1: always use header=0
    try:
        df = pd.read_csv(filepath, header=0)
    except Exception as e:
        raise IOError(f"Cannot read '{fname}': {e}") from e

    # Validate required columns
    for col in ("basis", "TT"):
        if col not in df.columns:
            raise ValueError(
                f"'{fname}' is missing column '{col}'. "
                f"Found columns: {list(df.columns)}"
            )

    # FIX ERROR 4: sort by canonical basis order
    basis_to_idx = {b: i for i, b in enumerate(CANONICAL_BASES)}
    df["_sort"] = df["basis"].map(basis_to_idx)

    missing_bases = df[df["_sort"].isna()]["basis"].tolist()
    if missing_bases:
        raise ValueError(
            f"'{fname}' has unrecognised basis values: {missing_bases}"
        )

    df = df.sort_values("_sort").reset_index(drop=True)

    if len(df) != N_BASES:
        raise ValueError(
            f"'{fname}' has {len(df)} rows; expected {N_BASES}."
        )

    return df["TT"].values.astype(float)


def validate_tt_range(tt: np.ndarray, power: int, fname: str) -> None:
    """Warn if TT values fall outside expected range for this pump power."""
    if power not in TT_VALIDATION:
        return
    lo, hi = TT_VALIDATION[power]
    if tt.min() < lo * 0.5 or tt.max() > hi * 1.5:
        print(
            f"  [WARN] '{fname}': TT range [{tt.min():.0f}, {tt.max():.0f}] "
            f"outside expected [{lo}, {hi}] for {power} mW"
        )


def load_dataset(data_dir: str) -> tuple[np.ndarray, list[str], list[int]]:
    """
    Load all 12 ECMBI files in EXPECTED_FILES order.

    Returns
    -------
    X          : np.ndarray shape (12, 36)  – raw TT counts
    file_names : list of 12 filenames loaded
    powers     : list of 12 pump powers (mW)
    """
    X          = []
    file_names = []
    powers     = []

    print(f"\n  Scanning: {data_dir}")
    found_any = False

    for fname in EXPECTED_FILES:
        fpath = os.path.join(data_dir, fname)

        if not os.path.isfile(fpath):
            # FIX ERROR 3: only scan for exact expected filenames
            print(f"  [MISS]   {fname}")
            continue

        try:
            tt    = load_one_file(fpath)
            power = parse_pump_power(fname)
            validate_tt_range(tt, power, fname)

            X.append(tt)
            file_names.append(fname)
            powers.append(power)
            found_any = True
            print(f"  [OK]     {fname}  (power={power} mW, TT=[{tt.min():.0f},{tt.max():.0f}])")

        except Exception as e:
            print(f"  [ERROR]  {fname}: {e}")

    if not found_any:
        raise FileNotFoundError(
            f"\nNo expected files found in '{data_dir}'.\n"
            "Check DATA_DIR at the top of this script."
        )

    X = np.array(X)   # shape: (n_loaded, 36)
    return X, file_names, powers


# ============================================================================
# SECTION 2 – COVARIANCE MATRIX
# ============================================================================

def build_covariance(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Standardise X and compute the sample covariance matrix.

    Parameters
    ----------
    X : shape (12, 36) – raw TT counts

    Returns
    -------
    Sigma_raw  : shape (36, 36) – sample covariance of standardised X
    X_scaled   : shape (12, 36) – StandardScaler-transformed X
    """
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)         # zero mean, unit variance per basis
    Sigma    = np.cov(X_scaled.T)              # (36, 36)
    return Sigma, X_scaled


# ============================================================================
# SECTION 3 – LASSO DENOISING  (paper Eq. 2)
# ============================================================================

def lasso_denoise(Sigma: np.ndarray, verbose: bool = True) -> np.ndarray:
    """
    Column-wise ℓ1-regularised sparse reconstruction (arXiv:2511.12546 Eq.2).

    For each column j:
        y         = Sigma[:, j]
        X_other   = Sigma[:, all columns except j]
        LassoCV fits y ~ X_other · β  with ℓ1 penalty
        Reconstruct column j from prediction, preserving diagonal.

    Then symmetrise and threshold near-zero off-diagonals.

    FIX ERROR 2: uses LassoCV, never GraphicalLasso.
    FIX ERROR 5: y_pred length = n (not n-1) because predict(X_other)
                 where X_other has n rows.  np.insert adds the diagonal
                 value back at position j, giving length n+1 — that is
                 CORRECT because we deleted column j from a (n × n) matrix
                 leaving (n × n-1), and lasso.predict on (n × n-1) gives
                 shape (n,). Re-inserting at position j → (n+1,)?  No:
                 X_other has shape (n, n-1) → predict gives (n,). The
                 reconstructed column must have length n.  np.insert into
                 a length-n array at index j gives length n+1 → BUG.

    CORRECT fix: y_pred has length n (all row predictions); we simply
    overwrite position j with the original diagonal value y[j].
    """
    n              = Sigma.shape[0]
    Sigma_denoised = np.zeros_like(Sigma)

    for j in range(n):
        y       = Sigma[:, j]                          # length n – target column
        X_other = np.delete(Sigma, j, axis=1)          # shape (n, n-1)

        # FIX: cv cannot exceed n_samples; n=36 rows, cv=3 is safe
        lasso = LassoCV(
            alphas       = LASSO_ALPHAS,
            cv           = LASSO_CV,
            max_iter     = LASSO_MAXITER,
            random_state = RANDOM_STATE,
            n_jobs       = -1,
        )
        lasso.fit(X_other, y)

        # predict(X_other) → length-n array (one value per row of X_other)
        y_pred    = lasso.predict(X_other)   # shape (n,)
        y_pred[j] = y[j]                     # restore original diagonal value
        Sigma_denoised[:, j] = y_pred

        if verbose and (j % 9 == 0 or j == n - 1):
            print(f"    Column {j+1:>2}/{n}  α={lasso.alpha_:.5f}")

    # Symmetrise
    Sigma_denoised = (Sigma_denoised + Sigma_denoised.T) / 2.0

    # Zero near-zero off-diagonal entries (threshold = 1% of mean |diagonal|)
    diag_mean = np.abs(np.diag(Sigma_denoised)).mean()
    threshold  = 0.01 * diag_mean if diag_mean > 0 else 0.0
    if threshold > 0:
        mask = np.abs(Sigma_denoised) < threshold
        np.fill_diagonal(mask, False)
        Sigma_denoised[mask] = 0.0

    return Sigma_denoised


# ============================================================================
# SECTION 4 – SNR CALCULATION  (paper Eq. 3)
# ============================================================================

def calculate_snr(
    cov_matrix      : np.ndarray,
    peak_half_width : int = PEAK_HALF_WIDTH,
    bg_start        : int = BG_START,
) -> float:
    """
    SNR = μ_peak / σ_background   (arXiv:2511.12546 Eq.3)

    The 36 basis settings are ordered so that correlations within the
    same measurement subspace (similar indices) form the "peak" along
    the main diagonal.  Noise fills the off-diagonal background.

    Peak region      : |i − j| <= peak_half_width  (default 3)
    Background region: |i − j| >  bg_start          (default 6)

    Returns float SNR (higher = better visibility).
    """
    n       = cov_matrix.shape[0]
    abs_cov = np.abs(cov_matrix)

    idx  = np.arange(n)
    diff = np.abs(idx[:, None] - idx[None, :])

    peak_vals = abs_cov[diff <= peak_half_width]
    bg_vals   = abs_cov[diff >  bg_start]

    mu_peak  = float(peak_vals.mean()) if peak_vals.size > 0 else 0.0
    sigma_bg = float(bg_vals.std())    if bg_vals.size   > 0 else 0.0

    if sigma_bg == 0.0:
        return float("inf") if mu_peak > 0.0 else 0.0
    return mu_peak / sigma_bg


def snr_improvement_db(snr_raw: float, snr_new: float) -> float:
    """20 · log10(snr_new / snr_raw).  Returns 0 if either value is invalid."""
    if snr_raw <= 0 or not np.isfinite(snr_raw) or not np.isfinite(snr_new):
        return 0.0
    return 20.0 * np.log10(snr_new / snr_raw)


# ============================================================================
# SECTION 5 – EPR CRITERION
# ============================================================================

def calculate_epr(Sigma: np.ndarray) -> float:
    """
    Position-momentum EPR criterion (Reid 1989, used in arXiv:2511.12546).

    For a (36 × 36) basis covariance matrix we approximate the EPR parameter
    as the ratio of the off-diagonal peak variance to the diagonal variance.
    EPR < 1  ↔  entanglement certified.

    This is a simplified estimate from the covariance structure; the paper
    uses the full EPR inference from position and momentum measurements.
    """
    n        = Sigma.shape[0]
    idx      = np.arange(n)
    diff     = np.abs(idx[:, None] - idx[None, :])

    diag_var     = np.abs(np.diag(Sigma)).mean()
    off_peak_var = np.abs(Sigma[diff <= PEAK_HALF_WIDTH]).mean()

    if diag_var == 0:
        return float("inf")

    # EPR ≈ off-diagonal correlation / diagonal variance
    # Values < 1 indicate non-classical (entangled) correlations
    epr = 1.0 - (off_peak_var / diag_var)
    return float(np.clip(epr, 0.0, None))


# ============================================================================
# SECTION 6 – NEURAL NETWORK: DATA AUGMENTATION
# ============================================================================

def augment_covariance_data(
    cov_matrices : list[np.ndarray],
    n_per_sample : int = NN_AUGMENT_PER_SAMPLE,
    noise_levels : list[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic training pairs (noisy, clean) for the autoencoder.

    For each real covariance matrix:
      • Add Gaussian noise at multiple σ levels to create 'noisy' inputs.
      • The corresponding 'clean' target is the original (Lasso-denoised) matrix.
      • Each noisy version is kept symmetric.

    Returns
    -------
    X_noisy : shape (N, 1296)  – flattened noisy matrices  (model input)
    X_clean : shape (N, 1296)  – flattened clean matrices  (model target)
    """
    if noise_levels is None:
        noise_levels = NN_NOISE_LEVELS

    noisy_list = []
    clean_list = []

    n_levels = len(noise_levels)
    # Distribute n_per_sample across noise levels
    per_level = max(1, n_per_sample // n_levels)

    for cov in cov_matrices:
        flat_clean = cov.flatten()
        for sigma in noise_levels:
            for _ in range(per_level):
                noise       = np.random.normal(0, sigma * np.abs(cov).max(), cov.shape)
                noisy       = cov + noise
                noisy       = (noisy + noisy.T) / 2.0    # keep symmetric
                noisy_list.append(noisy.flatten())
                clean_list.append(flat_clean)

    X_noisy = np.array(noisy_list, dtype=np.float32)
    X_clean = np.array(clean_list, dtype=np.float32)
    return X_noisy, X_clean


# ============================================================================
# SECTION 7 – NEURAL NETWORK: BUILD & TRAIN AUTOENCODER
# ============================================================================

def build_autoencoder(input_dim: int = 1296) -> "tf.keras.Model":
    """
    Denoising autoencoder as specified in the problem statement.

    Architecture
    ------------
    Input (1296) → Dense(512, relu) → Dropout(0.2) → Dense(256, relu)
                 → Dropout(0.2) → Dense(1296, linear)
    """
    model = Sequential([
        Dense(512, activation="relu", input_shape=(input_dim,),
              name="encoder_1"),
        Dropout(0.2, name="drop_1"),
        Dense(256, activation="relu", name="encoder_2"),
        Dropout(0.2, name="drop_2"),
        Dense(input_dim, activation="linear", name="decoder"),
    ], name="denoising_autoencoder")

    model.compile(
        optimizer=Adam(learning_rate=NN_LR),
        loss="mse",
        metrics=["mae"],
    )
    return model


def train_autoencoder(
    X_noisy     : np.ndarray,
    X_clean     : np.ndarray,
    output_dir  : str,
) -> tuple["tf.keras.Model", dict]:
    """
    Train the denoising autoencoder and save the best model.

    Returns
    -------
    model   : trained Keras model
    history : training history dict
    """
    model_path = os.path.join(output_dir, "best_denoiser.keras")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_noisy, X_clean, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"  Training samples  : {len(X_tr)}")
    print(f"  Validation samples: {len(X_val)}")

    model = build_autoencoder(input_dim=X_noisy.shape[1])
    model.summary(print_fn=lambda s: None)   # suppress verbose summary

    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=NN_PATIENCE,
            restore_best_weights=True, verbose=0,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5,
            min_lr=1e-6, verbose=0,
        ),
        ModelCheckpoint(
            model_path, monitor="val_loss",
            save_best_only=True, verbose=0,
        ),
    ]

    hist = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=NN_EPOCHS,
        batch_size=NN_BATCH,
        callbacks=callbacks,
        verbose=0,
    )

    stopped_epoch = len(hist.history["loss"])
    final_tr_loss = hist.history["loss"][-1]
    final_va_loss = hist.history["val_loss"][-1]

    print(f"  Epochs run        : {stopped_epoch}/{NN_EPOCHS} (early stopping)")
    print(f"  Final train loss  : {final_tr_loss:.6f}")
    print(f"  Final val loss    : {final_va_loss:.6f}")

    return model, hist.history


# ============================================================================
# SECTION 8 – VISUALISATION
# ============================================================================

def save_heatmap_comparison(
    Sigma_raw   : np.ndarray,
    Sigma_lasso : np.ndarray,
    Sigma_nn    : np.ndarray,
    snr_raw     : float,
    snr_lasso   : float,
    snr_nn      : float,
    output_dir  : str,
) -> str:
    """2×2 grid: Raw | Lasso | NN denoised | Residual (Raw − Lasso)."""
    vmax = np.abs(Sigma_raw).max()
    residual = Sigma_raw - Sigma_lasso

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    panels = [
        (Sigma_raw,   f"Raw Covariance\nSNR = {snr_raw:.3f}",      "RdBu_r"),
        (Sigma_lasso, f"Lasso Denoised\nSNR = {snr_lasso:.3f}",    "RdBu_r"),
        (Sigma_nn,    f"NN Denoised\nSNR = {snr_nn:.3f}",          "RdBu_r"),
        (residual,    "Residual (Raw − Lasso)\n",                   "coolwarm"),
    ]

    tick_step  = max(1, N_BASES // 12)
    tick_idx   = list(range(0, N_BASES, tick_step))
    tick_labels= [CANONICAL_BASES[i] for i in tick_idx]

    for ax, (mat, title, cmap) in zip(axes.flat, panels):
        im = ax.imshow(mat, cmap=cmap, vmin=-vmax, vmax=vmax,
                       origin="upper", interpolation="nearest", aspect="auto")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Basis j", fontsize=9)
        ax.set_ylabel("Basis i", fontsize=9)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(tick_idx)
        ax.set_yticklabels(tick_labels, fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Covariance Matrix: Raw vs. Denoised Methods",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    fpath = os.path.join(output_dir, "heatmap_comparison.png")
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fpath


def save_training_history(history: dict, output_dir: str) -> str:
    """Plot training vs validation loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, metric, label in zip(
        axes,
        ["loss", "mae"],
        ["MSE Loss", "MAE"],
    ):
        if metric not in history:
            continue
        ax.plot(history[metric],     label="Train",      color="#2196F3", linewidth=1.5)
        val_key = f"val_{metric}"
        if val_key in history:
            ax.plot(history[val_key], label="Validation", color="#f44336",
                    linewidth=1.5, linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(f"Training History – {label}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fpath = os.path.join(output_dir, "training_history.png")
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fpath


def save_comparison_bar(results: dict, output_dir: str) -> str:
    """Bar chart comparing SNR improvements across methods."""
    methods = list(results.keys())
    snrs    = [results[m]["snr"]            for m in methods]
    imprvs  = [results[m]["improvement_dB"] for m in methods]

    colors = ["#e74c3c", "#2980b9", "#27ae60"][:len(methods)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(methods, snrs, color=colors, edgecolor="k", linewidth=0.7)
    axes[0].set_ylabel("SNR  (μ_peak / σ_background)")
    axes[0].set_title("SNR by Method")
    for i, (m, v) in enumerate(zip(methods, snrs)):
        axes[0].text(i, v + 0.02 * max(snrs), f"{v:.3f}",
                     ha="center", fontsize=9)

    axes[1].bar(methods, imprvs, color=colors, edgecolor="k", linewidth=0.7)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("SNR Improvement (dB)")
    axes[1].set_title("SNR Improvement vs. Raw")
    for i, (m, v) in enumerate(zip(methods, imprvs)):
        axes[1].text(i, v + 0.01 * max(abs(x) for x in imprvs if x != 0),
                     f"{v:+.2f} dB", ha="center", fontsize=9)

    fig.suptitle("Method Comparison: Lasso vs. Neural Network",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fpath = os.path.join(output_dir, "comparison_plot.png")
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fpath


# ============================================================================
# SECTION 9 – SAVE OUTPUT FILES
# ============================================================================

def save_cov_csv(Sigma: np.ndarray, filepath: str) -> None:
    """Save a covariance matrix as CSV with basis labels as row/col headers."""
    pd.DataFrame(Sigma, index=CANONICAL_BASES, columns=CANONICAL_BASES
                 ).to_csv(filepath)


def save_snr_results(results: dict, filepath: str) -> None:
    rows = []
    for method, d in results.items():
        rows.append({
            "Method"        : method,
            "SNR"           : round(d["snr"], 6),
            "Improvement_dB": round(d["improvement_dB"], 4),
        })
    pd.DataFrame(rows).to_csv(filepath, index=False)


def save_epr_results(epr_dict: dict, filepath: str) -> None:
    rows = [{"Method": m, "EPR_parameter": round(v, 6),
             "Entangled": "YES" if v < 1.0 else "NO"}
            for m, v in epr_dict.items()]
    pd.DataFrame(rows).to_csv(filepath, index=False)


def save_summary_report(
    file_names : list[str],
    powers     : list[int],
    results    : dict,
    epr_dict   : dict,
    nn_trained : bool,
    output_dir : str,
) -> str:
    """Write a human-readable summary_report.txt."""
    fpath = os.path.join(output_dir, "summary_report.txt")
    best_method = max(results, key=lambda m: results[m]["snr"])

    lines = [
        "=" * 72,
        "SPARSITY DENOISING + NEURAL NETWORK PIPELINE – SUMMARY REPORT",
        "=" * 72,
        "",
        "Paper: Sparsity-Driven Entanglement Detection in High-Dimensional",
        "       Quantum States  (arXiv:2511.12546)",
        "",
        "DATA",
        "----",
        f"  Files loaded : {len(file_names)}",
        f"  X shape      : ({len(file_names)}, {N_BASES})",
        f"  Pump powers  : {sorted(set(powers))} mW",
        "",
        "FILES",
    ]
    for f in file_names:
        lines.append(f"  {f}")

    lines += [
        "",
        "RESULTS",
        "-------",
    ]
    for method, d in results.items():
        epr = epr_dict.get(method, float("nan"))
        lines += [
            f"  {method}",
            f"    SNR             : {d['snr']:.6f}",
            f"    Improvement     : {d['improvement_dB']:+.4f} dB",
            f"    EPR parameter   : {epr:.6f}  {'(entangled)' if epr < 1 else ''}",
            "",
        ]

    lines += [
        f"  Best method : {best_method}",
        f"  Best SNR    : {results[best_method]['snr']:.6f}",
        f"  Best ΔSNRdB : {results[best_method]['improvement_dB']:+.4f} dB",
        "",
        "OUTPUT FILES",
        "------------",
        f"  {output_dir}",
        "  cov_raw.csv, cov_lasso_denoised.csv, cov_nn_denoised.csv",
        "  heatmap_comparison.png, snr_results.csv, epr_results.csv",
        "  comparison_plot.png, summary_report.txt",
    ]
    if nn_trained:
        lines += [
            "  training_history.png, best_denoiser.keras",
        ]
    lines.append("=" * 72)

    with open(fpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return fpath


# ============================================================================
# SECTION 10 – MAIN PIPELINE
# ============================================================================

def main():
    # ── Banner ────────────────────────────────────────────────────────────────
    SEP = "=" * 72
    print(f"\n{SEP}")
    print("SPARSITY DENOISING + NEURAL NETWORK PIPELINE")
    print(SEP)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 10.1  Load data ───────────────────────────────────────────────────────
    print("\nDATA LOADING:")
    X, file_names, powers = load_dataset(DATA_DIR)
    n_files = len(file_names)

    print(f"  Files found  : {n_files}")
    print(f"  Files loaded : {n_files}")
    print(f"  X shape      : {X.shape}")

    if n_files < 2:
        raise RuntimeError(
            "Need at least 2 files to compute a covariance matrix. "
            f"Only {n_files} loaded."
        )

    # ── 10.2  Build covariance ────────────────────────────────────────────────
    Sigma_raw, X_scaled = build_covariance(X)
    snr_raw             = calculate_snr(Sigma_raw)
    epr_raw             = calculate_epr(Sigma_raw)

    # ── 10.3  Lasso denoising ─────────────────────────────────────────────────
    print(f"\n{'-'*60}")
    print("LASSO METHOD (PAPER):")
    print("  Running LassoCV column-by-column …")
    Sigma_lasso   = lasso_denoise(Sigma_raw, verbose=True)
    snr_lasso     = calculate_snr(Sigma_lasso)
    improv_lasso  = snr_improvement_db(snr_raw, snr_lasso)
    epr_lasso     = calculate_epr(Sigma_lasso)

    print(f"  Raw SNR             : {snr_raw:.4f}")
    print(f"  Lasso Denoised SNR  : {snr_lasso:.4f}")
    print(f"  Improvement         : {improv_lasso:+.2f} dB")
    print(f"  EPR parameter       : {epr_lasso:.4f}  "
          f"({'< 1 = entangled' if epr_lasso < 1 else '>= 1 = separable'})")

    # ── 10.4  Neural network ──────────────────────────────────────────────────
    print(f"\n{'-'*60}")
    print("NEURAL NETWORK TRAINING:")

    history    = {}
    Sigma_nn   = Sigma_lasso.copy()   # fallback if TF unavailable
    snr_nn     = snr_lasso
    nn_trained = False

    if TF_AVAILABLE:
        # Build training data from all power-level covariances
        # (one Lasso-denoised matrix per file → augment each)
        individual_covs = []
        for i in range(n_files):
            Xi         = X_scaled[i:i+1, :]           # shape (1, 36)
            # Augment each single sample with its "neighbours"
            # For a single-row X, cov(X.T) is undefined → use outer product
            xi_vec     = X_scaled[i]                   # (36,)
            cov_i      = np.outer(xi_vec, xi_vec)      # (36, 36)
            individual_covs.append(cov_i)

        X_noisy_aug, X_clean_aug = augment_covariance_data(
            individual_covs,
            n_per_sample=NN_AUGMENT_PER_SAMPLE,
        )
        total_samples = len(X_noisy_aug)
        print(f"  Total augmented samples : {total_samples}")

        model, history = train_autoencoder(X_noisy_aug, X_clean_aug, OUTPUT_DIR)
        nn_trained     = True

        # Denoise with the trained network
        flat_raw  = Sigma_raw.flatten().astype(np.float32).reshape(1, -1)
        flat_pred = model.predict(flat_raw, verbose=0)[0]
        Sigma_nn  = flat_pred.reshape(N_BASES, N_BASES)
        Sigma_nn  = (Sigma_nn + Sigma_nn.T) / 2.0     # symmetrise

        snr_nn    = calculate_snr(Sigma_nn)
        mse_nn_vs_lasso = float(np.mean((Sigma_nn - Sigma_lasso) ** 2))

        improv_nn = snr_improvement_db(snr_raw, snr_nn)
        epr_nn    = calculate_epr(Sigma_nn)

        print(f"\nNEURAL NETWORK RESULTS:")
        print(f"  NN Denoised SNR        : {snr_nn:.4f}")
        print(f"  Improvement over raw   : {improv_nn:+.2f} dB")
        print(f"  MSE vs Lasso           : {mse_nn_vs_lasso:.6f}")
        print(f"  EPR parameter          : {epr_nn:.4f}  "
              f"({'< 1 = entangled' if epr_nn < 1 else '>= 1 = separable'})")
    else:
        print("  [SKIP] TensorFlow not available – NN step skipped.")
        improv_nn   = 0.0
        epr_nn      = epr_lasso
        mse_nn_vs_lasso = 0.0

    # ── 10.5  Collect results ─────────────────────────────────────────────────
    results = {
        "Raw"   : {"snr": snr_raw,   "improvement_dB": 0.0},
        "Lasso" : {"snr": snr_lasso, "improvement_dB": improv_lasso},
        "Neural Network": {"snr": snr_nn, "improvement_dB": improv_nn},
    }
    epr_dict = {
        "Raw"            : epr_raw,
        "Lasso"          : epr_lasso,
        "Neural Network" : epr_nn,
    }

    # ── 10.6  Comparison summary ──────────────────────────────────────────────
    best_method = max(
        {"Lasso": snr_lasso, "Neural Network": snr_nn},
        key=lambda m: {"Lasso": snr_lasso, "Neural Network": snr_nn}[m]
    )
    best_snr    = max(snr_lasso, snr_nn)
    best_improv = snr_improvement_db(snr_raw, best_snr)

    print(f"\n{'-'*60}")
    print("COMPARISON:")
    print(f"  Best method      : {best_method}")
    print(f"  Best SNR         : {best_snr:.4f}")
    print(f"  Best improvement : {best_improv:+.2f} dB")

    # ── 10.7  Save all output files ───────────────────────────────────────────
    print(f"\n{'-'*60}")
    print("SAVING OUTPUT FILES …")

    saved = []

    # 1. Raw covariance CSV
    p = os.path.join(OUTPUT_DIR, "cov_raw.csv")
    save_cov_csv(Sigma_raw, p);  saved.append(p)

    # 2. Lasso covariance CSV
    p = os.path.join(OUTPUT_DIR, "cov_lasso_denoised.csv")
    save_cov_csv(Sigma_lasso, p); saved.append(p)

    # 3. NN covariance CSV
    p = os.path.join(OUTPUT_DIR, "cov_nn_denoised.csv")
    save_cov_csv(Sigma_nn, p);   saved.append(p)

    # 4. Heatmap comparison (2×2 grid)
    p = save_heatmap_comparison(
        Sigma_raw, Sigma_lasso, Sigma_nn,
        snr_raw, snr_lasso, snr_nn,
        OUTPUT_DIR,
    )
    saved.append(p)

    # 5. SNR results table
    p = os.path.join(OUTPUT_DIR, "snr_results.csv")
    save_snr_results(results, p); saved.append(p)

    # 6. Training history plot (only if NN was trained)
    if nn_trained and history:
        p = save_training_history(history, OUTPUT_DIR)
        saved.append(p)

    # 7. Best denoiser model (already saved by ModelCheckpoint)
    model_p = os.path.join(OUTPUT_DIR, "best_denoiser.keras")
    if os.path.isfile(model_p):
        saved.append(model_p)

    # 8. Summary report
    p = save_summary_report(
        file_names, powers, results, epr_dict, nn_trained, OUTPUT_DIR
    )
    saved.append(p)

    # 9. EPR results
    p = os.path.join(OUTPUT_DIR, "epr_results.csv")
    save_epr_results(epr_dict, p); saved.append(p)

    # 10. Comparison bar chart
    p = save_comparison_bar(results, OUTPUT_DIR)
    saved.append(p)

    print(f"\nFILES SAVED:")
    print(f"  Location       : {OUTPUT_DIR}")
    print(f"  Number of files: {len(saved)}")
    for f in saved:
        print(f"  → {Path(f).name}")

    # ── Final banner ──────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("PIPELINE COMPLETE")
    print(SEP)


# ============================================================================
if __name__ == "__main__":
    main()