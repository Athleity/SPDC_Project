

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['figure.dpi'] = 300

print("="*70)
print("GENERATING ALL THESIS FIGURES")
print("="*70)

# Load data
df = pd.read_csv('../03_Data/BBO_hybrid_data.csv')
wavelengths = df['Wavelength_nm'].values
n_o = df['n_o'].values
n_e = df['n_e'].values
delta_n = df['birefringence_delta_n'].values

# ================================================================
# FIGURE 1: Refractive Indices
# ================================================================
print("\n1. Creating Figure 1: Refractive Indices...")

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(wavelengths, n_o, 'bo-', linewidth=2, markersize=6, label='nₒ')
ax1.plot(wavelengths, n_e, 'rs-', linewidth=2, markersize=6, label='nₑ')
ax1.set_xlabel('Wavelength (nm)')
ax1.set_ylabel('Refractive Index')
ax1.set_title('(a) Refractive Indices')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(wavelengths, delta_n, 'g^-', linewidth=2, markersize=6, label='Δn = nₒ - nₑ')
ax2.set_xlabel('Wavelength (nm)')
ax2.set_ylabel('Birefringence')
ax2.set_title('(b) Birefringence')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Thesis_Fig1_Final.png', dpi=300)
plt.savefig('Thesis_Fig1_Final.pdf')
plt.close()
print("   ✓ Thesis_Fig1_Final.png/pdf")

# ================================================================
# FIGURE 2: Phase Matching
# ================================================================
print("\n2. Creating Figure 2: Phase Matching...")

pump = [400, 500, 532, 600, 700, 800, 900, 1000, 1064]
theta = [48.9, 38.5, 35.8, 33.2, 31.1, 29.2, 27.9, 26.8, 22.9]

fig2, ax = plt.subplots(figsize=(10, 6))
ax.plot(pump, theta, 'bo-', linewidth=2, markersize=8)
ax.set_xlabel('Pump Wavelength (nm)')
ax.set_ylabel('Phase Matching Angle (degrees)')
ax.set_title('BBO Phase Matching Curve')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Thesis_Fig2_Final.png', dpi=300)
plt.savefig('Thesis_Fig2_Final.pdf')
plt.close()
print("   ✓ Thesis_Fig2_Final.png/pdf")

# ================================================================
# FIGURE 3: Bell Inequality
# ================================================================
print("\n3. Creating Figure 3: Bell Inequality...")

fidelities = np.linspace(0.5, 1.0, 50)
S_values = 2.8284 * fidelities

fig3, ax = plt.subplots(figsize=(10, 6))
ax.plot(fidelities, S_values, 'b-', linewidth=2)
ax.axhline(y=2, color='r', linestyle='--', label='Classical limit (S=2)')
ax.axhline(y=2.828, color='g', linestyle=':', label='Quantum limit (2√2)')
ax.axvline(x=0.9622, color='purple', linestyle='--', label='Your fidelity')
ax.plot(0.9622, 2.7215, 'ro', markersize=10, label='Your prediction')
ax.set_xlabel('Fidelity')
ax.set_ylabel('Bell Parameter S')
ax.set_title('Bell Inequality Violation')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Thesis_Fig3_Final.png', dpi=300)
plt.savefig('Thesis_Fig3_Final.pdf')
plt.close()
print("   ✓ Thesis_Fig3_Final.png/pdf")

print("\n" + "="*70)
print("✅ ALL THESIS FIGURES GENERATED")
print("="*70)
print("\nFiles created:")
print("  - Thesis_Fig1_Final.png/pdf")
print("  - Thesis_Fig2_Final.png/pdf")
print("  - Thesis_Fig3_Final.png/pdf")
print("="*70)