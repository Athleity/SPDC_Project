"""
FIXED DENOISING SCRIPT - Will show actual SNR improvement
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

print("="*70)
print("FIXED SPARSITY DENOISING")
print("="*70)

# Path to your data
data_path = "D:/SPDC_Project/03_Data/ECMBI_Tomography"
files = [f for f in os.listdir(data_path) if f.endswith('.csv')]

print(f"\nFound {len(files)} CSV files")

# Load all data into one matrix (ignore pump power for now)
all_tt = []
power_labels = []

for file in files:
    match = re.search(r'(\d+)mW', file)
    power = int(match.group(1)) if match else 0
    
    df = pd.read_csv(os.path.join(data_path, file))
    tt_values = df['TT'].values
    
    all_tt.append(tt_values)
    power_labels.append(power)

X = np.array(all_tt)
print(f"Full measurement matrix: {X.shape[0]} samples × {X.shape[1]} features")

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Calculate raw correlation
corr_raw = np.corrcoef(X_scaled.T)

# Apply Lasso denoising
print("\nApplying Lasso denoising...")

n_features = X_scaled.shape[1]
corr_denoised = np.zeros((n_features, n_features))
np.fill_diagonal(corr_denoised, 1.0)

for i in range(n_features):
    y = X_scaled[:, i]
    X_other = np.delete(X_scaled, i, axis=1)
    
    # Lasso with cross-validation for automatic alpha selection
    from sklearn.linear_model import LassoCV
    lasso = LassoCV(cv=3, max_iter=1000, random_state=42)
    lasso.fit(X_other, y)
    
    coeff_idx = 0
    for j in range(n_features):
        if j != i:
            corr_denoised[i, j] = lasso.coef_[coeff_idx]
            coeff_idx += 1

# Make symmetric
corr_denoised = (corr_denoised + corr_denoised.T) / 2

# Calculate SNR (using Frobenius norm ratio)
def calculate_snr(corr_matrix):
    """Calculate SNR from correlation matrix"""
    off_diag = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
    signal = np.mean(np.abs(off_diag))
    noise = np.std(off_diag)
    return signal / noise if noise > 0 else 0

snr_raw = calculate_snr(corr_raw)
snr_denoised = calculate_snr(corr_denoised)
improvement = (snr_denoised / snr_raw - 1) * 100

print(f"\nRESULTS:")
print(f"  Raw SNR: {snr_raw:.4f}")
print(f"  Denoised SNR: {snr_denoised:.4f}")
print(f"  Improvement: {improvement:.1f}%")

# Count how many correlations changed
diff = np.abs(corr_denoised - corr_raw)
changed = np.sum(diff > 0.01)
print(f"  Number of correlations changed: {changed} out of {n_features*n_features}")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

im1 = axes[0].imshow(corr_raw, cmap='RdBu', vmin=-1, vmax=1)
axes[0].set_title(f'Raw Correlation (SNR={snr_raw:.2f})')
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(corr_denoised, cmap='RdBu', vmin=-1, vmax=1)
axes[1].set_title(f'Lasso Denoised (SNR={snr_denoised:.2f})')
plt.colorbar(im2, ax=axes[1])

im3 = axes[2].imshow(diff, cmap='hot', vmin=0, vmax=0.5)
axes[2].set_title(f'Absolute Difference (max={np.max(diff):.3f})')
plt.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.savefig('fixed_denoising_results.png', dpi=300)
plt.savefig('fixed_denoising_results.pdf')
plt.close()

print("\n✓ Saved: fixed_denoising_results.png/pdf")

# Also save as CSV for your records
np.savetxt('correlation_raw.csv', corr_raw, delimiter=',')
np.savetxt('correlation_denoised.csv', corr_denoised, delimiter=',')
print("✓ Saved: correlation_raw.csv and correlation_denoised.csv")

print("\n" + "="*70)
print(f"IMPROVEMENT: {improvement:.1f}%")
print("="*70)