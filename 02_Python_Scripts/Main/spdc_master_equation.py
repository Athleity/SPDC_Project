
import numpy as np
import matplotlib.pyplot as plt
import math

print("="*70)
print("SPDC MASTER EQUATION - FIXED")
print("="*70)

# Parameters
pump_power_mW = 500
pump_wavelength_nm = 355
crystal_length_mm = 3.0
repetition_rate_MHz = 80
pulse_duration_fs = 100
pump_waist_um = 25

collection_efficiency = 0.12
detector_efficiency = 0.35

# Constants
c = 3e8
epsilon0 = 8.85e-12

# BBO parameters
d_eff = 2.0e-12
n_pump = 1.6
n_signal = 1.6
n_idler = 1.6

# Calculations
repetition_rate_Hz = repetition_rate_MHz * 1e6
pulse_duration_s = pulse_duration_fs * 1e-15
pump_waist_m = pump_waist_um * 1e-6
pump_area = np.pi * pump_waist_m**2

pump_power_W = pump_power_mW / 1000
pulse_energy = pump_power_W / repetition_rate_Hz
peak_power = pulse_energy / pulse_duration_s
pump_intensity = peak_power / pump_area

L = crystal_length_mm / 1000
kappa = (2 * np.pi * d_eff / (n_pump * pump_wavelength_nm * 1e-9)) * np.sqrt(2 * pump_intensity / (epsilon0 * c * n_signal * n_idler))
gain = kappa * L

if gain < 1:
    pairs_per_pulse = (gain**2) / 4
else:
    pairs_per_pulse = (np.sinh(gain)**2) / 4

print(f"Pairs per pulse: {pairs_per_pulse:.4e}")

# Photon number distribution - as a line plot (not bars) for large mean
mean = pairs_per_pulse
std = np.sqrt(mean)
print(f"Mean: {mean:.2e}, Std: {std:.2e}")

# Plot around the mean ± 3 sigma
n_min = max(0, int(mean - 4*std))
n_max = int(mean + 4*std)
n_vals = list(range(n_min, n_max + 1))

probs = []
for n in n_vals:
    # Use log to avoid overflow
    log_prob = -mean + n * np.log(mean) - math.lgamma(n + 1)
    probs.append(np.exp(log_prob))

# Crystal length scan
lengths = np.linspace(0.5, 5, 50)
pairs_len = []
for Lmm in lengths:
    Lm = Lmm / 1000
    g = kappa * Lm
    if g < 1:
        pairs_len.append((g**2) / 4)
    else:
        pairs_len.append((np.sinh(g)**2) / 4)

# Pump power scan
powers = np.linspace(0, 1000, 50)
pairs_pow = []
detected = []
eff = collection_efficiency * detector_efficiency

for PmW in powers:
    PW = PmW / 1000
    pulse_E = PW / repetition_rate_Hz
    peak_P = pulse_E / pulse_duration_s
    I = peak_P / pump_area
    kp = (2 * np.pi * d_eff / (n_pump * pump_wavelength_nm * 1e-9)) * np.sqrt(2 * I / (epsilon0 * c * n_signal * n_idler))
    g = kp * L
    if g < 1:
        pp = (g**2) / 4
    else:
        pp = (np.sinh(g)**2) / 4
    pairs_pow.append(pp)
    detected.append(pp * repetition_rate_Hz * eff**2)

# Reset matplotlib
plt.rcParams.update(plt.rcParamsDefault)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1 - Photon number distribution (line plot)
axes[0, 0].plot(n_vals, probs, 'b-', linewidth=1.5)
axes[0, 0].set_xlabel('Number of Pairs per Pulse')
axes[0, 0].set_ylabel('Probability')
axes[0, 0].set_title(f'Photon Number Distribution (Mean = {mean:.2e})')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xlim(n_min, n_max)

# Plot 2
axes[0, 1].plot(lengths, pairs_len, 'b-', linewidth=2)
axes[0, 1].axvline(x=crystal_length_mm, color='r', linestyle='--', label=f'Your crystal: {crystal_length_mm} mm')
axes[0, 1].set_xlabel('Crystal Length (mm)')
axes[0, 1].set_ylabel('Pairs per Pulse')
axes[0, 1].set_title('Pairs vs Crystal Length')
axes[0, 1].set_yscale('log')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3
axes[1, 0].plot(powers, pairs_pow, 'b-', linewidth=2)
axes[1, 0].axvline(x=pump_power_mW, color='r', linestyle='--', label=f'Your power: {pump_power_mW} mW')
axes[1, 0].set_xlabel('Pump Power (mW)')
axes[1, 0].set_ylabel('Pairs per Pulse')
axes[1, 0].set_title('Pairs vs Pump Power')
axes[1, 0].set_yscale('log')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4
axes[1, 1].plot(powers, detected, 'purple', linewidth=2)
axes[1, 1].axvline(x=pump_power_mW, color='r', linestyle='--', label=f'Your power: {pump_power_mW} mW')
axes[1, 1].set_xlabel('Pump Power (mW)')
axes[1, 1].set_ylabel('Detected Rate (Hz)')
axes[1, 1].set_title('Detected Coincidence Rate')
axes[1, 1].set_yscale('log')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('spdc_master_equation_result.png', dpi=300, bbox_inches='tight')
plt.savefig('spdc_master_equation_result.pdf', bbox_inches='tight')
plt.close()

print("✓ Saved: spdc_master_equation_result.png/pdf")
print("="*70)