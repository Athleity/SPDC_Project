

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

print("="*70)
print("PhD-LEVEL: EXPERIMENTAL DATA FITTER")
print("="*70)

# ================================================================
# CREATE SAMPLE DATA (REPLACE WITH YOUR LAB DATA)
# ================================================================

# Theoretical model: Coincidence rate = A * (Pump_Power)^2 + B
def coincidence_model(pump_power_mW, A, B):
    return A * (pump_power_mW / 100)**2 + B

# Generate sample data (replace with your actual lab measurements)
pump_powers = np.array([50, 100, 150, 200, 300, 400, 500])
coincidence_measured = np.array([0.05, 0.22, 0.48, 0.85, 1.95, 3.48, 5.42])
coincidence_error = np.array([0.02, 0.03, 0.04, 0.05, 0.08, 0.12, 0.15])

print("\n" + "="*70)
print("SAMPLE LAB DATA (REPLACE WITH YOURS)")
print("="*70)
print(f"{'Power (mW)':<12} {'Coincidence (Hz)':<18} {'Error (Hz)':<12}")
print("-"*45)
for i in range(len(pump_powers)):
    print(f"{pump_powers[i]:<12} {coincidence_measured[i]:<18.2f} {coincidence_error[i]:<12.2f}")

# ================================================================
# FIT THE DATA
# ================================================================

print("\n" + "="*70)
print("FITTING DATA TO THEORETICAL MODEL")
print("="*70)

popt, pcov = curve_fit(coincidence_model, pump_powers, coincidence_measured, 
                        sigma=coincidence_error, absolute_sigma=True)
A_fit, B_fit = popt
A_err, B_err = np.sqrt(np.diag(pcov))

print(f"\nFit parameters:")
print(f"  A = {A_fit:.4f} ± {A_err:.4f}")
print(f"  B = {B_fit:.4f} ± {B_err:.4f}")

# Calculate chi-squared
coincidence_fit = coincidence_model(pump_powers, A_fit, B_fit)
chi_squared = np.sum(((coincidence_measured - coincidence_fit) / coincidence_error)**2)
dof = len(pump_powers) - 2
chi_squared_reduced = chi_squared / dof

print(f"\nStatistics:")
print(f"  Chi-squared = {chi_squared:.2f}")
print(f"  Degrees of freedom = {dof}")
print(f"  Reduced chi-squared = {chi_squared_reduced:.3f}")

if chi_squared_reduced < 1:
    print("  ✓ Good fit (overestimated errors)")
elif chi_squared_reduced < 2:
    print("  ✓ Acceptable fit")
else:
    print("  ✗ Poor fit (check model or data)")

# ================================================================
# CALCULATE QUANTUM METRICS FROM DATA
# ================================================================

# From your quantum tomography
fidelity = 0.9622
concurrence = 0.9245
bell_S = 2.7215

print("\n" + "="*70)
print("QUANTUM METRICS FROM YOUR DATA")
print("="*70)
print(f"  Fidelity: {fidelity:.4f}")
print(f"  Concurrence: {concurrence:.4f}")
print(f"  Bell parameter S: {bell_S:.4f}")

# ================================================================
# GENERATE PLOTS
# ================================================================

print("\nGenerating plots...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Coincidence vs pump power with fit
pump_smooth = np.linspace(0, 550, 100)
coincidence_smooth = coincidence_model(pump_smooth, A_fit, B_fit)

axes[0, 0].errorbar(pump_powers, coincidence_measured, yerr=coincidence_error, 
                     fmt='ro', markersize=8, capsize=3, label='Experimental data')
axes[0, 0].plot(pump_smooth, coincidence_smooth, 'b-', linewidth=2, label=f'Fit: y = {A_fit:.3f}(x/100)² + {B_fit:.3f}')
axes[0, 0].set_xlabel('Pump Power (mW)')
axes[0, 0].set_ylabel('Coincidence Rate (Hz)')
axes[0, 0].set_title('Coincidence Rate vs Pump Power')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Residuals
residuals = coincidence_measured - coincidence_fit
axes[0, 1].errorbar(pump_powers, residuals, yerr=coincidence_error, 
                     fmt='bo', markersize=8, capsize=3)
axes[0, 1].axhline(y=0, color='r', linestyle='--')
axes[0, 1].set_xlabel('Pump Power (mW)')
axes[0, 1].set_ylabel('Residuals (Hz)')
axes[0, 1].set_title('Fit Residuals')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Quantum metrics bar chart
metrics = ['Fidelity', 'Concurrence', 'Bell S / 3']
values = [fidelity, concurrence, bell_S / 3]
thresholds = [0.89, 0.5, 0.667]
colors_plot = ['green' if v > t else 'red' for v, t in zip(values, thresholds)]

axes[1, 0].bar(metrics, values, color=colors_plot, alpha=0.7)
axes[1, 0].axhline(y=0.89, color='b', linestyle='--', label='Fidelity threshold')
axes[1, 0].axhline(y=0.5, color='g', linestyle='--', label='Concurrence threshold')
axes[1, 0].axhline(y=0.667, color='r', linestyle='--', label='Bell violation threshold')
axes[1, 0].set_ylabel('Value')
axes[1, 0].set_title('Quantum Metrics')
axes[1, 0].legend()
axes[1, 0].set_ylim(0, 1.1)
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Chi-squared visualization
chi_values = np.linspace(0, 20, 200)
chi_prob = np.exp(-chi_values/2) * chi_values**(dof/2 - 1)
chi_norm = chi_prob / np.max(chi_prob)

axes[1, 1].plot(chi_values, chi_norm, 'b-', linewidth=2)
axes[1, 1].axvline(x=chi_squared, color='r', linestyle='--', linewidth=2, label=f'Your χ² = {chi_squared:.1f}')
axes[1, 1].fill_between(chi_values, 0, chi_norm, where=chi_values >= chi_squared, 
                         color='red', alpha=0.3, label='Worse fit region')
axes[1, 1].set_xlabel('Chi-squared')
axes[1, 1].set_ylabel('Probability')
axes[1, 1].set_title(f'Chi-squared Distribution (dof = {dof})')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('experimental_fit_results.png', dpi=300, bbox_inches='tight')
plt.savefig('experimental_fit_results.pdf', bbox_inches='tight')
plt.close()

print("✓ Saved: experimental_fit_results.png/pdf")

# Save fit data
fit_data = np.column_stack([pump_powers, coincidence_measured, coincidence_error, coincidence_fit, residuals])
np.savetxt('fit_data.csv', fit_data, delimiter=',', 
           header='Pump_power_mW,Coincidence_Hz,Coincidence_error_Hz,Fit_Hz,Residuals_Hz')

print("✓ Saved: fit_data.csv")

print("\n" + "="*70)
print("✅ EXPERIMENTAL DATA FITTER COMPLETE")
print("="*70)
print("""
TO USE WITH YOUR REAL LAB DATA:
1. Edit the 'pump_powers', 'coincidence_measured', 'coincidence_error' arrays
2. Replace sample data with your actual measurements
3. Run the script again
""")
print("="*70)