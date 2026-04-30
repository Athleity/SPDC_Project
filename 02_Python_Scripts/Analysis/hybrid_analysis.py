

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("="*60)
print("BBO CRYSTAL DATA ANALYSIS")
print("="*60)
print("\nThis script will help you enter data from SNLO once.")
print("Then it automatically creates all graphs and calculations.")
print()

# ================================================================
# STEP 1: ENTER YOUR DATA FROM SNLO (Do this once)
# ================================================================

print("Please enter the data from SNLO Ref. Ind. function")
print("(θ = 90°, T = 293 K)")
print("-"*50)

wavelengths = []
n_o_values = []
n_e_values = []

# Pre-defined wavelengths
default_wavelengths = [400, 500, 583, 600, 700, 800, 900, 1000, 1064]

for wl in default_wavelengths:
    print(f"\nWavelength: {wl} nm")
    
    # Get n_o
    while True:
        try:
            n_o = float(input(f"  Enter n_o at {wl} nm: "))
            break
        except:
            print("  Please enter a valid number")
    
    # Get n_e
    while True:
        try:
            n_e = float(input(f"  Enter n_e at {wl} nm: "))
            break
        except:
            print("  Please enter a valid number")
    
    wavelengths.append(wl)
    n_o_values.append(n_o)
    n_e_values.append(n_e)

# ================================================================
# STEP 2: CALCULATIONS
# ================================================================

delta_n = [n_o_values[i] - n_e_values[i] for i in range(len(wavelengths))]

# Create DataFrame
df = pd.DataFrame({
    'Wavelength_nm': wavelengths,
    'n_o': n_o_values,
    'n_e': n_e_values,
    'birefringence_delta_n': delta_n
})

# Save to CSV
df.to_csv('BBO_hybrid_data.csv', index=False)
print("\n" + "="*60)
print("✓ Data saved to 'BBO_hybrid_data.csv'")
print("="*60)

# Print summary
print("\nData Summary:")
print("-"*60)
print(f"{'λ (nm)':<12} {'n_o':<12} {'n_e':<12} {'Δn':<12}")
print("-"*60)
for i in range(len(wavelengths)):
    print(f"{wavelengths[i]:<12} {n_o_values[i]:<12.4f} {n_e_values[i]:<12.4f} {delta_n[i]:<12.4f}")
print("-"*60)

# ================================================================
# STEP 3: CREATE PLOTS
# ================================================================

print("\n📊 Generating plots...")

# Figure 1: Refractive indices
plt.figure(figsize=(12, 8))
plt.plot(wavelengths, n_o_values, 'bo-', linewidth=2.5, markersize=8,
         markerfacecolor='white', markeredgewidth=2, label='nₒ (ordinary)')
plt.plot(wavelengths, n_e_values, 'rs-', linewidth=2.5, markersize=8,
         markerfacecolor='white', markeredgewidth=2, label='nₑ (extraordinary)')
plt.plot(wavelengths, delta_n, 'g^--', linewidth=2, markersize=8,
         markerfacecolor='white', markeredgewidth=2, label='Δn (birefringence)')

plt.xlabel('Wavelength (nm)', fontsize=14, fontweight='bold')
plt.ylabel('Refractive Index', fontsize=14, fontweight='bold')
plt.title('BBO Crystal: Refractive Indices vs Wavelength\n(θ = 90°, T = 293 K)', 
          fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# Highlight SPDC wavelengths
plt.axvline(x=583, color='purple', linestyle=':', alpha=0.8, linewidth=2)
plt.text(585, min(n_e_values)-0.01, 'Signal (583 nm)', fontsize=11, color='purple')
plt.axvline(x=900, color='orange', linestyle=':', alpha=0.8, linewidth=2)
plt.text(905, min(n_e_values)-0.01, 'Idler (900 nm)', fontsize=11, color='orange')

plt.savefig('BBO_hybrid_plot.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('BBO_hybrid_plot.pdf', bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Plot saved to 'BBO_hybrid_plot.png' and 'BBO_hybrid_plot.pdf'")

# ================================================================
# STEP 4: COMPENSATOR CALCULATION
# ================================================================

print("\n" + "="*60)
print("BABINET COMPENSATOR CALCULATION")
print("="*60)

# Find or interpolate values at 583 nm
if 583 in wavelengths:
    idx = wavelengths.index(583)
    n_o_583 = n_o_values[idx]
    n_e_583 = n_e_values[idx]
else:
    # Interpolate
    n_o_583 = np.interp(583, wavelengths, n_o_values)
    n_e_583 = np.interp(583, wavelengths, n_e_values)

delta_n_583 = n_o_583 - n_e_583

# BBO crystal thickness (CHANGE THIS to your actual crystal)
crystal_thickness_mm = 1.0

# Calculate phase shift
lambda_mm = 583 / 1_000_000
phi = (2 * np.pi * delta_n_583 * crystal_thickness_mm) / lambda_mm
residual = phi % (2 * np.pi)
compensator_deg = np.degrees(residual)

print(f"\nSignal wavelength: 583 nm")
print(f"n_o = {n_o_583:.4f}, n_e = {n_e_583:.4f}")
print(f"Δn = {delta_n_583:.4f}")
print(f"Crystal thickness: {crystal_thickness_mm} mm")
print(f"\nResidual phase: {residual:.4f} rad ({np.degrees(residual):.2f}°)")
print(f"\n🔬 Set Babinet compensator to: {compensator_deg:.1f}°")

print("\n" + "="*60)
print("✅ ANALYSIS COMPLETE!")
print("="*60)
print("Files created:")
print("  - BBO_hybrid_data.csv")
print("  - BBO_hybrid_plot.png")
print("  - BBO_hybrid_plot.pdf")
print("="*60)