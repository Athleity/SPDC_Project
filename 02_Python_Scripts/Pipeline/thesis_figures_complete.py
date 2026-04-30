

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ================================================================
# THESIS-QUALITY STYLE SETTINGS
# ================================================================

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 6

print("="*70)
print("PHASE 5: GENERATING THESIS-READY FIGURES")
print("="*70)

# Load your data
df = pd.read_csv('BBO_hybrid_data.csv')
wavelengths = df['Wavelength_nm'].values
n_o = df['n_o'].values
n_e = df['n_e'].values
delta_n = df['birefringence_delta_n'].values

# ================================================================
# FIGURE 1: REFRACTIVE INDICES WITH ERROR BARS
# ================================================================

print("\n📊 FIGURE 1: Refractive Indices with Error Bars")

unc_n = 0.0021  # from error analysis
unc_dn = unc_n * np.sqrt(2)

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Subplot 1a: n_o and n_e
ax1.errorbar(wavelengths, n_o, yerr=unc_n, fmt='o-', color='#1f77b4',
             capsize=3, capthick=1, elinewidth=1, markersize=6,
             label=r'$n_o$ (ordinary)')
ax1.errorbar(wavelengths, n_e, yerr=unc_n, fmt='s-', color='#ff7f0e',
             capsize=3, capthick=1, elinewidth=1, markersize=6,
             label=r'$n_e$ (extraordinary)')
ax1.set_xlabel('Wavelength (nm)', fontsize=12)
ax1.set_ylabel('Refractive Index', fontsize=12)
ax1.set_title('(a) Refractive Indices', fontsize=12)
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3, linestyle='--')

# Subplot 1b: Birefringence
ax2.errorbar(wavelengths, delta_n, yerr=unc_dn, fmt='^-', color='#2ca02c',
             capsize=3, capthick=1, elinewidth=1, markersize=6,
             label=r'$\Delta n = n_o - n_e$')
ax2.set_xlabel('Wavelength (nm)', fontsize=12)
ax2.set_ylabel('Birefringence', fontsize=12)
ax2.set_title('(b) Birefringence', fontsize=12)
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('Thesis_Fig1_Refractive_Indices.png', dpi=300)
plt.savefig('Thesis_Fig1_Refractive_Indices.pdf')
plt.close()
print("   ✓ Thesis_Fig1_Refractive_Indices.png/pdf")

# ================================================================
# FIGURE 2: PHASE MATCHING CURVE
# ================================================================

print("\n📊 FIGURE 2: Phase Matching Curve")

pump_wavelengths = [400, 450, 500, 532, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1064]
theta_pm = [48.9, 43.2, 38.5, 35.8, 34.5, 33.2, 32.0, 31.1, 30.0, 29.2, 28.5, 27.9, 27.3, 26.8, 22.9]
walkoff_mrad = [5.2, 4.8, 4.4, 4.1, 4.0, 3.8, 3.6, 3.5, 3.4, 3.3, 3.2, 3.1, 3.0, 2.9, 2.5]

fig2, ax1 = plt.subplots(figsize=(10, 6))

color1 = 'blue'
ax1.set_xlabel('Pump Wavelength (nm)', fontsize=12)
ax1.set_ylabel('Phase Matching Angle θ (degrees)', fontsize=12, color=color1)
ax1.plot(pump_wavelengths, theta_pm, 'o-', color=color1, linewidth=2, markersize=6, label='Phase Matching Angle')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(20, 55)

ax2 = ax1.twinx()
color2 = 'red'
ax2.set_ylabel('Walkoff Angle (mrad)', fontsize=12, color=color2)
ax2.plot(pump_wavelengths, walkoff_mrad, 's-', color=color2, linewidth=2, markersize=6, label='Walkoff Angle')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(2, 6)

plt.title('BBO Type I Phase Matching and Walkoff', fontsize=14)
ax1.grid(True, alpha=0.3, linestyle='--')

# Add legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

plt.tight_layout()
plt.savefig('Thesis_Fig2_Phase_Matching.png', dpi=300)
plt.savefig('Thesis_Fig2_Phase_Matching.pdf')
plt.close()
print("   ✓ Thesis_Fig2_Phase_Matching.png/pdf")

# ================================================================
# FIGURE 3: SPDC SIMULATION RESULTS
# ================================================================

print("\n📊 FIGURE 3: SPDC Simulation")

# Constants
c = 3e8
epsilon0 = 8.854e-12
d_eff = 2.0e-12
L_m = 0.001
n_eff = 1.6
spot_area = np.pi * (50e-6)**2
repetition_rate = 80e6
pulse_duration = 100e-15
total_efficiency = 0.042

pump_powers_mW = np.linspace(0, 200, 100)
coincidence_rates = []

for P_mW in pump_powers_mW:
    P_W = P_mW / 1000
    pulse_energy = P_W / repetition_rate
    peak_power = pulse_energy / pulse_duration
    intensity = peak_power / spot_area
    gain = 2 * np.pi * d_eff * np.sqrt(intensity / (epsilon0 * c * n_eff**3)) * L_m
    pairs_per_pulse = gain**2 / 4
    rate = pairs_per_pulse * repetition_rate * total_efficiency**2
    coincidence_rates.append(rate)

fig3, ax = plt.subplots(figsize=(10, 6))
ax.plot(pump_powers_mW, coincidence_rates, 'b-', linewidth=2)
ax.set_xlabel('Pump Power (mW)', fontsize=12)
ax.set_ylabel('Coincidence Rate (Hz)', fontsize=12)
ax.set_title('SPDC Coincidence Rate vs Pump Power', fontsize=14)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_yscale('log')

plt.tight_layout()
plt.savefig('Thesis_Fig3_SPDC_Simulation.png', dpi=300)
plt.savefig('Thesis_Fig3_SPDC_Simulation.pdf')
plt.close()
print("   ✓ Thesis_Fig3_SPDC_Simulation.png/pdf")

# ================================================================
# FIGURE 4: COMPENSATOR CALIBRATION
# ================================================================

print("\n📊 FIGURE 4: Compensator Calibration")

displacements_um = np.linspace(0, 50, 500)
compensator_birefringence = 0.009
lambda_nm = 583
phases_deg = (displacements_um / 1000) * (360 / (lambda_nm * 1e-6 / compensator_birefringence))

target_phase = 77.5
target_displacement = (target_phase / 360) * (lambda_nm * 1e-6 / compensator_birefringence) * 1000

fig4, ax = plt.subplots(figsize=(10, 6))
ax.plot(displacements_um, phases_deg, 'b-', linewidth=2)
ax.axhline(y=target_phase, color='r', linestyle='--', linewidth=1.5, label=f'Target: {target_phase}°')
ax.axvline(x=target_displacement, color='g', linestyle='--', linewidth=1.5, label=f'Displacement: {target_displacement:.1f} μm')
ax.set_xlabel('Compensator Displacement (micrometers)', fontsize=12)
ax.set_ylabel('Phase Retardation (degrees)', fontsize=12)
ax.set_title('Babinet Compensator Calibration Curve (583 nm)', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('Thesis_Fig4_Compensator_Calibration.png', dpi=300)
plt.savefig('Thesis_Fig4_Compensator_Calibration.pdf')
plt.close()
print("   ✓ Thesis_Fig4_Compensator_Calibration.png/pdf")

# ================================================================
# FIGURE 5: ALL CRYSTALS COMPARISON (BBO vs LBO vs KTP)
# ================================================================

print("\n📊 FIGURE 5: Crystal Comparison")

crystal_data = {
    'BBO': {'d_eff': 2.0, 'damage': 10, 'walkoff': 3.5, 'color': '#1f77b4'},
    'LBO': {'d_eff': 0.85, 'damage': 25, 'walkoff': 0.5, 'color': '#ff7f0e'},
    'KTP': {'d_eff': 3.5, 'damage': 15, 'walkoff': 0.2, 'color': '#2ca02c'}
}

crystals = list(crystal_data.keys())
d_eff_values = [crystal_data[c]['d_eff'] for c in crystals]
damage_values = [crystal_data[c]['damage'] for c in crystals]
walkoff_values = [crystal_data[c]['walkoff'] for c in crystals]

x = np.arange(len(crystals))
width = 0.25

fig5, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width, d_eff_values, width, label='d_eff (pm/V)', color='#1f77b4')
bars2 = ax.bar(x, damage_values, width, label='Damage Threshold (GW/cm²)', color='#ff7f0e')
bars3 = ax.bar(x + width, walkoff_values, width, label='Walkoff (degrees)', color='#2ca02c')

ax.set_xlabel('Crystal', fontsize=12)
ax.set_ylabel('Value', fontsize=12)
ax.set_title('Nonlinear Crystal Performance Comparison', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(crystals)
ax.legend()

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('Thesis_Fig5_Crystal_Comparison.png', dpi=300)
plt.savefig('Thesis_Fig5_Crystal_Comparison.pdf')
plt.close()
print("   ✓ Thesis_Fig5_Crystal_Comparison.png/pdf")

# ================================================================
# SUMMARY TABLE (TXT format for thesis appendix)
# ================================================================

print("\n📊 Generating Summary Table")

with open('Thesis_Data_Summary.txt', 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("BBO CRYSTAL DATA SUMMARY FOR THESIS\n")
    f.write("="*70 + "\n\n")
    
    f.write("Table 1: Refractive Indices of BBO (theta = 90 deg, T = 293 K)\n")
    f.write("-"*60 + "\n")
    f.write(f"{'lambda (nm)':<12} {'n_o':<12} {'n_e':<12} {'Delta n':<12}\n")
    f.write("-"*60 + "\n")
    
    for i in range(len(wavelengths)):
        f.write(f"{wavelengths[i]:<12} {n_o[i]:<12.4f} {n_e[i]:<12.4f} {delta_n[i]:<12.4f}\n")
    
    f.write("-"*60 + "\n\n")
    
    f.write("Table 2: Phase Matching Data for BBO (Type I, o+o->e)\n")
    f.write("-"*60 + "\n")
    f.write(f"{'pump lambda (nm)':<16} {'theta_PM (deg)':<16} {'Walkoff (mrad)':<16}\n")
    f.write("-"*60 + "\n")
    
    for i in range(len(pump_wavelengths)):
        f.write(f"{pump_wavelengths[i]:<16} {theta_pm[i]:<16.1f} {walkoff_mrad[i]:<16.2f}\n")
    
    f.write("-"*60 + "\n\n")
    
    f.write("Table 3: Compensator Calculation Results\n")
    f.write("-"*60 + "\n")
    f.write(f"Signal wavelength: 583 nm\n")
    f.write(f"n_o = {np.interp(583, wavelengths, n_o):.4f}\n")
    f.write(f"n_e = {np.interp(583, wavelengths, n_e):.4f}\n")
    f.write(f"Delta n = {np.interp(583, wavelengths, delta_n):.4f}\n")
    f.write(f"Compensator setting: 77.5 degrees\n")
    f.write("-"*60 + "\n")

print("   ✓ Thesis_Data_Summary.txt")

# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n" + "="*70)
print("PHASE 5 COMPLETE!")
print("="*70)
print("\n📁 FILES GENERATED FOR YOUR THESIS:")
print("-"*50)
print("FIGURES:")
print("  📊 Thesis_Fig1_Refractive_Indices.png/pdf")
print("  📊 Thesis_Fig2_Phase_Matching.png/pdf")
print("  📊 Thesis_Fig3_SPDC_Simulation.png/pdf")
print("  📊 Thesis_Fig4_Compensator_Calibration.png/pdf")
print("  📊 Thesis_Fig5_Crystal_Comparison.png/pdf")
print("\nDATA:")
print("  📄 Thesis_Data_Summary.txt")
print("\n" + "="*70)
print("✅ All thesis-ready figures have been generated!")
print("📁 Check your D:\\SPDC_Project folder")
print("="*70)