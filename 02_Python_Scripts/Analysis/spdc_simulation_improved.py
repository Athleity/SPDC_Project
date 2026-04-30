

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

print("="*70)
print("IMPROVED SPDC SIMULATION FOR BBO CRYSTAL")
print("="*70)

# ================================================================
# CONSTANTS
# ================================================================

c = 299792458
epsilon0 = 8.854e-12

# ================================================================
# EXPERIMENTAL PARAMETERS
# ================================================================

pump_wavelength_nm = 355
pump_power_mW = 500
pulse_duration_fs = 100
repetition_rate_MHz = 80
pump_waist_um = 25

crystal_length_mm = 3.0
theta_deg = 30.5

collection_efficiency = 0.12
detector_efficiency = 0.35
coincidence_window_ns = 1.0

lambda_signal_nm = 583
lambda_idler_nm = 900

print("\n" + "="*70)
print("SIMULATION PARAMETERS")
print("="*70)
print(f"Pump: {pump_wavelength_nm} nm, {pump_power_mW} mW, {pulse_duration_fs} fs")
print(f"BBO: {crystal_length_mm} mm, theta = {theta_deg}°")

# ================================================================
# BBO SELLMEIER EQUATIONS
# ================================================================

def sellmeier_o(lambda_um):
    lambda_sq = lambda_um**2
    return np.sqrt(2.7359 + 0.01878/(lambda_sq - 0.01822) - 0.01354*lambda_sq)

def sellmeier_e(lambda_um):
    lambda_sq = lambda_um**2
    return np.sqrt(2.3753 + 0.01224/(lambda_sq - 0.01667) - 0.01516*lambda_sq)

def n_eff_extraordinary(lambda_um, theta_deg):
    theta_rad = np.radians(theta_deg)
    n_o = sellmeier_o(lambda_um)
    n_e = sellmeier_e(lambda_um)
    cos2 = np.cos(theta_rad)**2
    sin2 = np.sin(theta_rad)**2
    return 1 / np.sqrt(cos2/n_o**2 + sin2/n_e**2)

# ================================================================
# PHASE MATCHING
# ================================================================

def phase_mismatch(lambda_s_nm, lambda_i_nm, theta_deg):
    lambda_p_um = pump_wavelength_nm / 1000
    lambda_s_um = lambda_s_nm / 1000
    lambda_i_um = lambda_i_nm / 1000
    
    n_p = n_eff_extraordinary(lambda_p_um, theta_deg)
    n_s = sellmeier_o(lambda_s_um)
    n_i = sellmeier_o(lambda_i_um)
    
    k_p = 2 * np.pi * n_p / lambda_p_um
    k_s = 2 * np.pi * n_s / lambda_s_um
    k_i = 2 * np.pi * n_i / lambda_i_um
    
    return k_p - k_s - k_i

# ================================================================
# SPDC CALCULATION
# ================================================================

def calculate_coincidence_rate(pump_power_mW):
    
    # Pump parameters
    pump_power_W = pump_power_mW / 1000
    pulse_duration_s = pulse_duration_fs * 1e-15
    repetition_rate_Hz = repetition_rate_MHz * 1e6
    
    # Pump intensity
    pump_waist_m = pump_waist_um * 1e-6
    pump_area = np.pi * pump_waist_m**2
    pulse_energy = pump_power_W / repetition_rate_Hz
    peak_power = pulse_energy / pulse_duration_s
    pump_intensity = peak_power / pump_area
    
    # BBO parameters
    d_eff = 2.0e-12  # m/V
    L = crystal_length_mm / 1000
    
    # Refractive indices
    lambda_p_um = pump_wavelength_nm / 1000
    lambda_s_um = lambda_signal_nm / 1000
    lambda_i_um = lambda_idler_nm / 1000
    
    n_p = n_eff_extraordinary(lambda_p_um, theta_deg)
    n_s = sellmeier_o(lambda_s_um)
    n_i = sellmeier_o(lambda_i_um)
    
    # Phase mismatch
    dk = phase_mismatch(lambda_signal_nm, lambda_idler_nm, theta_deg)
    
    # Sinc^2 factor
    if abs(dk) < 1e-9:
        sinc2 = 1.0
    else:
        sinc2 = (np.sin(dk * L / 2) / (dk * L / 2))**2
    
    # Gain coefficient
    gamma = (2 * np.pi * d_eff / (n_p * pump_wavelength_nm * 1e-9)) * np.sqrt(2 * pump_intensity / (epsilon0 * c * n_s * n_i))
    
    # Gain
    gain = gamma * L * np.sqrt(sinc2)
    
    # Pairs per pulse (corrected for high gain)
    if gain < 1:
        pairs_per_pulse = (gain**2) / 4
    else:
        # For high gain, use hyperbolic sine approximation
        pairs_per_pulse = (np.sinh(gain)**2) / 4
    
    # Walk-off reduction
    walkoff_deg = 3.5
    walkoff_rad = np.radians(walkoff_deg)
    displacement = walkoff_rad * L
    overlap = np.exp(-(displacement / pump_waist_m)**2 / 2)
    pairs_per_pulse *= overlap
    
    # Detection efficiency
    total_efficiency = collection_efficiency * detector_efficiency
    
    # Coincidence rate
    coincidence_rate = pairs_per_pulse * repetition_rate_Hz * total_efficiency**2
    
    # Background
    dark_counts = 100
    accidental = 2 * dark_counts * coincidence_window_ns * 1e-9 * repetition_rate_Hz
    
    return coincidence_rate, accidental, gain, pairs_per_pulse, pump_intensity, sinc2

# ================================================================
# PHASE MATCHING BANDWIDTH
# ================================================================

def phase_matching_bandwidth(theta_deg):
    wavelengths = np.linspace(550, 650, 500)
    sinc2_values = []
    
    for lam_s in wavelengths:
        lam_i = 1/(1/pump_wavelength_nm - 1/lam_s)
        if lam_i > 0 and lam_i < 2000:
            dk = phase_mismatch(lam_s, lam_i, theta_deg)
            L = crystal_length_mm / 1000
            if abs(dk) < 1e-9:
                sinc2 = 1
            else:
                sinc2 = (np.sin(dk * L / 2) / (dk * L / 2))**2
            sinc2_values.append(sinc2)
        else:
            sinc2_values.append(0)
    
    return wavelengths, np.array(sinc2_values)

# ================================================================
# RUN SIMULATION
# ================================================================

print("\n" + "="*70)
print("Phase Matching Bandwidth")
print("="*70)

wavelengths_scan, sinc2 = phase_matching_bandwidth(theta_deg)
above_half = np.where(sinc2 > 0.5)[0]
if len(above_half) > 0:
    bandwidth_nm = wavelengths_scan[above_half[-1]] - wavelengths_scan[above_half[0]]
else:
    bandwidth_nm = 0
print(f"Phase matching bandwidth (FWHM): {bandwidth_nm:.1f} nm")

print("\n" + "="*70)
print("Coincidence Rate Prediction")
print("="*70)

rate, accidental, gain, pairs, intensity, sinc2_val = calculate_coincidence_rate(pump_power_mW)

print(f"Pump intensity: {intensity/1e12:.3f} TW/m²")
print(f"Sinc² factor: {sinc2_val:.4f}")
print(f"Gain parameter: {gain:.4f}")
print(f"Pairs per pulse: {pairs:.6e}")
print(f"Coincidence rate: {rate:.6f} Hz")
print(f"Accidental background: {accidental:.2f} Hz")
if accidental > 0:
    print(f"Signal-to-noise ratio: {rate/accidental:.2f}")

# ================================================================
# GENERATE PLOTS
# ================================================================

print("\n" + "="*70)
print("Generating Plots")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Phase matching function
axes[0, 0].plot(wavelengths_scan, sinc2, 'b-', linewidth=2)
axes[0, 0].axhline(y=0.5, color='r', linestyle='--', label='FWHM')
axes[0, 0].set_xlabel('Signal Wavelength (nm)')
axes[0, 0].set_ylabel('Relative Efficiency')
axes[0, 0].set_title(f'Phase Matching (FWHM = {bandwidth_nm:.1f} nm)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Coincidence rate vs pump power
pump_powers = np.linspace(0, 500, 100)
rates = []
for P in pump_powers:
    r, _, _, _, _, _ = calculate_coincidence_rate(P)
    rates.append(r)

axes[0, 1].plot(pump_powers, rates, 'b-', linewidth=2)
axes[0, 1].set_xlabel('Pump Power (mW)')
axes[0, 1].set_ylabel('Coincidence Rate (Hz)')
axes[0, 1].set_title('Coincidence Rate vs Pump Power')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_yscale('log')

# Plot 3: Pairs per pulse vs crystal length
lengths = np.linspace(0.5, 5, 50)
pairs_vs_length = []
for L in lengths:
    old_length = crystal_length_mm
    crystal_length_mm = L
    _, _, _, p, _, _ = calculate_coincidence_rate(pump_power_mW)
    pairs_vs_length.append(p)
    crystal_length_mm = old_length

axes[1, 0].plot(lengths, pairs_vs_length, 'b-', linewidth=2)
axes[1, 0].set_xlabel('Crystal Length (mm)')
axes[1, 0].set_ylabel('Pairs per pulse')
axes[1, 0].set_title('SPDC Pairs vs Crystal Length')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_yscale('log')

# Plot 4: Coincidence vs waist
waists = np.linspace(10, 100, 50)
rates_vs_waist = []
for w in waists:
    old_waist = pump_waist_um
    pump_waist_um = w
    r, _, _, _, _, _ = calculate_coincidence_rate(pump_power_mW)
    rates_vs_waist.append(r)
    pump_waist_um = old_waist

axes[1, 1].plot(waists, rates_vs_waist, 'b-', linewidth=2)
axes[1, 1].set_xlabel('Beam Waist (μm)')
axes[1, 1].set_ylabel('Coincidence Rate (Hz)')
axes[1, 1].set_title('Rate vs Focus Size')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('SPDC_simulation_improved.png', dpi=300, bbox_inches='tight')
plt.savefig('SPDC_simulation_improved.pdf', bbox_inches='tight')
plt.close()

print("✓ Saved: SPDC_simulation_improved.png/pdf")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"""
Predicted Performance (500 mW, 3 mm):
  - Coincidence rate: {rate:.4f} Hz
  - Pairs per pulse: {pairs:.4e}
  - Bandwidth: {bandwidth_nm:.1f} nm
  - Gain: {gain:.4f}

Lab Settings:
  - BBO angle: {theta_deg}°
  - Signal: {lambda_signal_nm} nm
  - Idler: {lambda_idler_nm} nm
""")
print("="*70)