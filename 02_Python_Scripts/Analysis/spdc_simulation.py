import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

print("="*70)
print("PHASE 4: SPDC SIMULATION FOR BBO CRYSTAL")
print("="*70)

# ================================================================
# INPUT PARAMETERS (CHANGE THESE FOR YOUR SETUP)
# ================================================================

pump_wavelength_nm = 355
pump_power_mW = 100
pulse_duration_fs = 100
repetition_rate_MHz = 80

crystal_length_mm = 1.0
theta_deg = 30.5

collection_efficiency = 0.12
detector_efficiency = 0.35

print("\n" + "="*70)
print("SIMULATION PARAMETERS")
print("="*70)
print(f"Pump wavelength: {pump_wavelength_nm} nm")
print(f"Pump power: {pump_power_mW} mW")
print(f"Pulse duration: {pulse_duration_fs} fs")
print(f"Repetition rate: {repetition_rate_MHz} MHz")
print(f"BBO length: {crystal_length_mm} mm")

# ================================================================
# STEP 1: SIGNAL AND IDLER WAVELENGTHS
# ================================================================

print("\n" + "="*70)
print("STEP 1: Signal and Idler Wavelengths")
print("="*70)

lambda_signal_nm = 583
lambda_idler_nm = 900

# Energy conservation check
check = 1/pump_wavelength_nm - (1/lambda_signal_nm + 1/lambda_idler_nm)
print(f"Signal: {lambda_signal_nm} nm, Idler: {lambda_idler_nm} nm")
print(f"Energy conservation check: {check*1e6:.2f} μm⁻¹")

# ================================================================
# STEP 2: PHASE MATCHING BANDWIDTH
# ================================================================

print("\n" + "="*70)
print("STEP 2: Phase Matching Bandwidth")
print("="*70)

angular_bandwidth_mrad_cm = 0.85
angular_bandwidth_mrad = angular_bandwidth_mrad_cm / (crystal_length_mm/10)
angular_bandwidth_deg = angular_bandwidth_mrad * 0.0573

spectral_bandwidth_nm_cm = 0.45
spectral_bandwidth_nm = spectral_bandwidth_nm_cm / (crystal_length_mm/10)

print(f"Angular bandwidth: {angular_bandwidth_deg:.3f}°")
print(f"Spectral bandwidth: {spectral_bandwidth_nm:.2f} nm")

# ================================================================
# STEP 3: SPDC EFFICIENCY
# ================================================================

print("\n" + "="*70)
print("STEP 3: SPDC Efficiency")
print("="*70)

# Constants
c = 3e8
epsilon0 = 8.854e-12

# BBO parameters
d_eff = 2.0e-12  # m/V

# Pump parameters
spot_radius_um = 50
spot_area_m2 = np.pi * (spot_radius_um * 1e-6)**2

# Calculate pump intensity (W/m²)
pump_power_W = pump_power_mW / 1000
pulse_energy_J = pump_power_W / (repetition_rate_MHz * 1e6)
peak_power_W = pulse_energy_J / (pulse_duration_fs * 1e-15)
pump_intensity_W_m2 = peak_power_W / spot_area_m2

print(f"Spot radius: {spot_radius_um} μm")
print(f"Pulse energy: {pulse_energy_J*1e9:.2f} nJ")
print(f"Peak power: {peak_power_W/1e6:.2f} MW")
print(f"Peak intensity: {pump_intensity_W_m2/1e12:.3f} TW/m²")

# Gain calculation
L_m = crystal_length_mm / 1000
n_eff = 1.6  # average refractive index
gain = 2 * np.pi * d_eff * np.sqrt(pump_intensity_W_m2 / (epsilon0 * c * n_eff**3)) * L_m

# Pairs per pulse (for SPDC, gain is small)
pairs_per_pulse = gain**2 / 4

print(f"Gain parameter: {gain:.4f}")
print(f"Pairs per pulse: {pairs_per_pulse:.6f}")

# ================================================================
# STEP 4: COINCIDENCE RATE
# ================================================================

print("\n" + "="*70)
print("STEP 4: Detected Coincidence Rate")
print("="*70)

total_efficiency = collection_efficiency * detector_efficiency
coincidence_rate = pairs_per_pulse * repetition_rate_MHz * 1e6 * total_efficiency**2

print(f"Detection efficiency: {total_efficiency*100:.1f}%")
print(f"Predicted coincidence rate: {coincidence_rate:.2f} Hz")

# ================================================================
# STEP 5: GENERATE PLOTS
# ================================================================

print("\n" + "="*70)
print("STEP 5: Generating Tuning Curves")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Signal wavelength vs crystal angle
angles = np.linspace(20, 40, 100)
theta_ref = 30.5
lambda_ref = 583
signal_wavelengths = lambda_ref * (np.sin(np.radians(theta_ref)) / np.sin(np.radians(angles)))**2

axes[0, 0].plot(angles, signal_wavelengths, 'b-', linewidth=2)
axes[0, 0].axhline(y=583, color='r', linestyle='--', label='Target (583 nm)')
axes[0, 0].axvline(x=theta_ref, color='g', linestyle='--', label=f'θ = {theta_ref}°')
axes[0, 0].set_xlabel('Crystal Angle (degrees)', fontsize=12)
axes[0, 0].set_ylabel('Signal Wavelength (nm)', fontsize=12)
axes[0, 0].set_title('SPDC Tuning Curve', fontsize=14)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Phase matching function
deltak = np.linspace(-10, 10, 500) * 1e3
L_m = crystal_length_mm / 1000
phase_matching = np.sinc(deltak * L_m / (2 * np.pi))**2

axes[0, 1].plot(deltak/1e3, phase_matching, 'b-', linewidth=2)
axes[0, 1].set_xlabel('Phase Mismatch Δk (×10³ m⁻¹)', fontsize=12)
axes[0, 1].set_ylabel('Relative Efficiency', fontsize=12)
axes[0, 1].set_title('Phase Matching Function', fontsize=14)
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Coincidence rate vs pump power
pump_powers = np.linspace(0, 200, 100)
coincidence_rates = []

for P_mW in pump_powers:
    P_W = P_mW / 1000
    pulse_E = P_W / (repetition_rate_MHz * 1e6)
    peak_P = pulse_E / (pulse_duration_fs * 1e-15)
    I = peak_P / spot_area_m2
    g = 2 * np.pi * d_eff * np.sqrt(I / (epsilon0 * c * n_eff**3)) * L_m
    pp = g**2 / 4
    rate = pp * repetition_rate_MHz * 1e6 * total_efficiency**2
    coincidence_rates.append(rate)

axes[1, 0].plot(pump_powers, coincidence_rates, 'b-', linewidth=2)
axes[1, 0].axvline(x=pump_power_mW, color='r', linestyle='--', label=f'Your pump: {pump_power_mW} mW')
axes[1, 0].set_xlabel('Pump Power (mW)', fontsize=12)
axes[1, 0].set_ylabel('Coincidence Rate (Hz)', fontsize=12)
axes[1, 0].set_title('Coincidence Rate vs Pump Power', fontsize=14)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Coincidence rate vs crystal length
lengths = np.linspace(0.1, 5, 100)
coincidence_vs_length = []

for L_mm in lengths:
    L = L_mm / 1000
    g = 2 * np.pi * d_eff * np.sqrt(pump_intensity_W_m2 / (epsilon0 * c * n_eff**3)) * L
    pp = g**2 / 4
    rate = pp * repetition_rate_MHz * 1e6 * total_efficiency**2
    coincidence_vs_length.append(rate)

axes[1, 1].plot(lengths, coincidence_vs_length, 'b-', linewidth=2)
axes[1, 1].axvline(x=crystal_length_mm, color='r', linestyle='--', label=f'Your crystal: {crystal_length_mm} mm')
axes[1, 1].set_xlabel('Crystal Length (mm)', fontsize=12)
axes[1, 1].set_ylabel('Coincidence Rate (Hz)', fontsize=12)
axes[1, 1].set_title('Coincidence Rate vs Crystal Length', fontsize=14)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('SPDC_simulation.png', dpi=300, bbox_inches='tight')
plt.savefig('SPDC_simulation.pdf', bbox_inches='tight')
plt.close()

print("✓ SPDC simulation plots saved to 'SPDC_simulation.png' and 'SPDC_simulation.pdf'")

# ================================================================
# SUMMARY
# ================================================================

print("\n" + "="*70)
print("PHASE 4 COMPLETE - SUMMARY")
print("="*70)
print(f"""
Predicted Performance:
  - Pairs per pulse: {pairs_per_pulse:.6f}
  - Coincidence rate: {coincidence_rate:.2f} Hz

Optimization Tips:
  1. Increase pump power (quadratic scaling)
  2. Optimize crystal length (~1-2 mm for BBO)
  3. Improve collection/detection efficiency
  4. Use tighter focusing (smaller spot size)

Files Created:
  - SPDC_simulation.png
  - SPDC_simulation.pdf
""")
print("="*70)