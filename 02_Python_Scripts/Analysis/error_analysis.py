

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("="*60)
print("PHASE 3: ERROR ANALYSIS FOR BBO CRYSTAL")
print("="*60)

# Load your data
df = pd.read_csv('BBO_hybrid_data.csv')
wavelengths = df['Wavelength_nm'].values
n_o = df['n_o'].values
n_e = df['n_e'].values
delta_n = df['birefringence_delta_n'].values

# ================================================================
# STEP 1: ESTIMATE MEASUREMENT UNCERTAINTIES
# ================================================================

print("\n" + "="*60)
print("STEP 1: Uncertainty Estimation")
print("="*60)

uncertainty_n = 0.0005
uncertainty_wavelength = 0.1

print(f"Refractive index uncertainty: ±{uncertainty_n}")
print(f"Wavelength uncertainty: ±{uncertainty_wavelength} nm")

uncertainty_delta_n = np.sqrt(uncertainty_n**2 + uncertainty_n**2)
print(f"Birefringence uncertainty: ±{uncertainty_delta_n:.4f}")

# ================================================================
# STEP 2: POLYNOMIAL FIT
# ================================================================

print("\n" + "="*60)
print("STEP 2: Polynomial Fitting")
print("="*60)

coeff_o = np.polyfit(wavelengths, n_o, 3)
n_o_fit = np.polyval(coeff_o, wavelengths)
residuals_o = n_o - n_o_fit
rms_error_o = np.sqrt(np.mean(residuals_o**2))

coeff_e = np.polyfit(wavelengths, n_e, 3)
n_e_fit = np.polyval(coeff_e, wavelengths)
residuals_e = n_e - n_e_fit
rms_error_e = np.sqrt(np.mean(residuals_e**2))

print(f"\nn_o fit RMS error: {rms_error_o:.5f}")
print(f"n_e fit RMS error: {rms_error_e:.5f}")

# Use the larger of the two for uncertainty
unc_n = max(rms_error_o, rms_error_e, uncertainty_n)
print(f"\nUsing uncertainty: ±{unc_n:.4f}")

# ================================================================
# STEP 3: VALUES AT 583 nm
# ================================================================

print("\n" + "="*60)
print("STEP 3: Values at Signal Wavelength (583 nm)")
print("="*60)

n_o_583 = np.polyval(coeff_o, 583)
n_e_583 = np.polyval(coeff_e, 583)
delta_n_583 = n_o_583 - n_e_583

print(f"n_o(583) = {n_o_583:.4f} ± {unc_n:.4f}")
print(f"n_e(583) = {n_e_583:.4f} ± {unc_n:.4f}")
print(f"Δn(583) = {delta_n_583:.4f} ± {unc_n*np.sqrt(2):.4f}")

# ================================================================
# STEP 4: COMPENSATOR CALCULATION WITH PROPER ERROR
# ================================================================

print("\n" + "="*60)
print("STEP 4: Babinet Compensator Calculation")
print("="*60)

crystal_thickness = 1.0  # mm (CHANGE THIS)
unc_thickness = 0.05  # ±0.05 mm

lambda_nm = 583
lambda_mm = lambda_nm / 1_000_000

# Calculate phase
phase = (2 * np.pi * delta_n_583 * crystal_thickness) / lambda_mm

# Residual phase (modulo 2π)
residual = phase % (2 * np.pi)
compensator_deg = np.degrees(residual)

print(f"Crystal thickness: {crystal_thickness} ± {unc_thickness} mm")
print(f"Total phase: {phase:.1f} rad")
print(f"Residual phase (mod 2π): {residual:.4f} rad ({compensator_deg:.1f}°)")

# ================================================================
# STEP 5: ERROR PROPAGATION (ON RESIDUAL)
# ================================================================

print("\n" + "="*60)
print("STEP 5: Error Propagation")
print("="*60)

# Sensitivity factors
d_phase_d_delta_n = (2 * np.pi * crystal_thickness) / lambda_mm
d_phase_d_thickness = (2 * np.pi * delta_n_583) / lambda_mm

# Uncertainty in delta_n (from n_o and n_e)
unc_delta_n = unc_n * np.sqrt(2)

# Uncertainty in phase
unc_phase_total = np.sqrt(
    (d_phase_d_delta_n * unc_delta_n)**2 +
    (d_phase_d_thickness * unc_thickness)**2
)

# For modulo 2π, the uncertainty in residual is the same as uncertainty in phase
# BUT we need to consider that small errors can cause large jumps near 0 or 2π
unc_residual = unc_phase_total % (2 * np.pi)

# Convert to degrees
unc_compensator_deg = np.degrees(unc_residual)

print(f"Sensitivity d(phase)/d(Δn) = {d_phase_d_delta_n:.1f} rad per unit Δn")
print(f"Sensitivity d(phase)/d(thickness) = {d_phase_d_thickness:.1f} rad/mm")
print(f"\nPhase uncertainty: ±{unc_phase_total:.2f} rad")
print(f"Residual phase uncertainty: ±{unc_residual:.4f} rad")
print(f"\n🔬 Final compensator setting: {compensator_deg:.1f}° ± {unc_compensator_deg:.1f}°")

# ================================================================
# STEP 6: CREATE ERROR BAR PLOT
# ================================================================

print("\n" + "="*60)
print("STEP 6: Generating Error Bar Plot")
print("="*60)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: n_o and n_e with error bars
ax1.errorbar(wavelengths, n_o, yerr=unc_n, fmt='bo-', capsize=3,
             linewidth=2, markersize=6, label='n_o (ordinary)')
ax1.errorbar(wavelengths, n_e, yerr=unc_n, fmt='rs-', capsize=3,
             linewidth=2, markersize=6, label='n_e (extraordinary)')
ax1.set_xlabel('Wavelength (nm)', fontsize=12)
ax1.set_ylabel('Refractive Index', fontsize=12)
ax1.set_title('BBO Refractive Indices with Uncertainty', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Birefringence with error bars
unc_delta_n_arr = unc_n * np.sqrt(2) * np.ones_like(delta_n)
ax2.errorbar(wavelengths, delta_n, yerr=unc_delta_n_arr, fmt='g^-', capsize=3,
             linewidth=2, markersize=6, label='Δn = n_o - n_e')
ax2.set_xlabel('Wavelength (nm)', fontsize=12)
ax2.set_ylabel('Birefringence Δn', fontsize=12)
ax2.set_title('BBO Birefringence with Uncertainty', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('BBO_error_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig('BBO_error_analysis.pdf', bbox_inches='tight')
plt.close()

print("✓ Error analysis plot saved to 'BBO_error_analysis.png' and 'BBO_error_analysis.pdf'")

# ================================================================
# SUMMARY
# ================================================================

print("\n" + "="*60)
print("PHASE 3 COMPLETE - SUMMARY")
print("="*60)
print(f"""
Measurement Uncertainties:
  - Refractive index: ±{unc_n:.4f}
  - Crystal thickness: ±{unc_thickness} mm

Compensator Setting (at 583 nm):
  - Value: {compensator_deg:.1f}°
  - Uncertainty: ±{unc_compensator_deg:.1f}°
  - Recommended range: {compensator_deg - unc_compensator_deg:.1f}° to {compensator_deg + unc_compensator_deg:.1f}°

Files Created:
  - BBO_error_analysis.png
  - BBO_error_analysis.pdf
""")
print("="*60)