

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

print("="*60)
print("GENERATING REALISTIC GRAPHS WITH NOISE")
print("="*60)

# Load your data
df = pd.read_csv('../03_Data/BBO_hybrid_data.csv')
wavelengths = df['Wavelength_nm'].values
n_o = df['n_o'].values
n_e = df['n_e'].values

# ================================================================
# CREATE REALISTIC DATA WITH NOISE
# ================================================================

# Add Poisson noise to simulate real measurements
np.random.seed(42)  # reproducible

# Measurement uncertainty from your lab setup
# Typical: ±0.001 for refractive index measurements
noise_level = 0.001

n_o_exp = n_o + np.random.normal(0, noise_level, len(n_o))
n_e_exp = n_e + np.random.normal(0, noise_level, len(n_e))

# Calculate error bars (standard deviation from multiple measurements)
error_bars = noise_level * np.ones(len(wavelengths))

# ================================================================
# FIGURE: THEORY vs EXPERIMENT (SIMULATED)
# ================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: n_o with experimental points
ax1.plot(wavelengths, n_o, 'b-', linewidth=2, label='Theory (SNLO)')
ax1.errorbar(wavelengths, n_o_exp, yerr=error_bars, fmt='bo', 
             capsize=3, markersize=5, label='Simulated Experiment', alpha=0.7)
ax1.set_xlabel('Wavelength (nm)')
ax1.set_ylabel('n_o (ordinary index)')
ax1.set_title('(a) Ordinary Refractive Index: Theory vs Experiment')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: n_e with experimental points
ax2.plot(wavelengths, n_e, 'r-', linewidth=2, label='Theory (SNLO)')
ax2.errorbar(wavelengths, n_e_exp, yerr=error_bars, fmt='rs', 
             capsize=3, markersize=5, label='Simulated Experiment', alpha=0.7)
ax2.set_xlabel('Wavelength (nm)')
ax2.set_ylabel('n_e (extraordinary index)')
ax2.set_title('(b) Extraordinary Refractive Index: Theory vs Experiment')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Theory_vs_Experiment_Simulated.png', dpi=300)
plt.savefig('Theory_vs_Experiment_Simulated.pdf')
plt.close()

print("✓ Created: Theory_vs_Experiment_Simulated.png/pdf")

# ================================================================
# RESIDUAL PLOT (Difference between theory and experiment)
# ================================================================

fig, ax = plt.subplots(figsize=(10, 5))

residuals_o = n_o_exp - n_o
residuals_e = n_e_exp - n_e

ax.axhline(y=0, color='k', linestyle='-', linewidth=1)
ax.errorbar(wavelengths, residuals_o, yerr=error_bars, fmt='bo', 
            capsize=3, markersize=5, label='n_o residuals', alpha=0.7)
ax.errorbar(wavelengths, residuals_e, yerr=error_bars, fmt='rs', 
            capsize=3, markersize=5, label='n_e residuals', alpha=0.7)
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Residual (Experiment - Theory)')
ax.set_title('Fit Residuals: Theory vs Simulated Experiment')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Residuals_Plot.png', dpi=300)
plt.savefig('Residuals_Plot.pdf')
plt.close()

print("✓ Created: Residuals_Plot.png/pdf")

# ================================================================
# SPDC COINCIDENCE WITH REALISTIC NOISE
# ================================================================

# Simulate pump power scan with noise
pump_powers = np.linspace(0, 200, 50)
coincidence_ideal = 0.01 * (pump_powers / 100)**2  # quadratic

# Add Poisson noise (photon counting statistics)
coincidence_exp = coincidence_ideal + np.random.poisson(np.sqrt(coincidence_ideal)) / 10
coincidence_error = np.sqrt(coincidence_ideal) / 10  # sqrt(N) error

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(pump_powers, coincidence_ideal, 'b-', linewidth=2, label='Theory (quadratic)')
ax.errorbar(pump_powers[::5], coincidence_exp[::5], yerr=coincidence_error[::5], 
            fmt='ro', capsize=3, markersize=6, label='Simulated Data', alpha=0.7)
ax.set_xlabel('Pump Power (mW)')
ax.set_ylabel('Coincidence Rate (Hz)')
ax.set_title('SPDC Coincidence Rate: Theory vs Simulated Experiment')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

plt.tight_layout()
plt.savefig('SPDC_Coincidence_With_Noise.png', dpi=300)
plt.savefig('SPDC_Coincidence_With_Noise.pdf')
plt.close()

print("✓ Created: SPDC_Coincidence_With_Noise.png/pdf")

print("\n" + "="*60)
print("REALISTIC GRAPHS GENERATED")
print("="*60)
print("Files created:")
print("  - Theory_vs_Experiment_Simulated.png/pdf")
print("  - Residuals_Plot.png/pdf")
print("  - SPDC_Coincidence_With_Noise.png/pdf")
print("\nNOTE: These are SIMULATED with added noise.")
print("Replace with YOUR actual experimental data when available.")
print("="*60)