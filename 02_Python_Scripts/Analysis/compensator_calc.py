"""
Babinet (Soleil-Babinet) Compensator Calculator for BBO SPDC Experiment
Based on SNLO Ref. Ind. measurements
Author: Your Name
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_SCRIPTS_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_SCRIPTS_DIR))
import config

config.ensure_dirs(config.THESIS_FIG_DIR, config.RESULTS_CSV_DIR)

# ============================================
# Load crystal + compensator parameters from config.py
# ============================================

n_o_583 = config.BBO_N_O_583
n_e_583 = config.BBO_N_E_583
n_o_900 = config.BBO_N_O_900
n_e_900 = config.BBO_N_E_900
crystal_thickness_mm = config.BBO_CRYSTAL_THICKNESS_MM

n_o_583_unc = config.BBO_N_O_583_UNC
n_e_583_unc = config.BBO_N_E_583_UNC
n_o_900_unc = config.BBO_N_O_900_UNC
n_e_900_unc = config.BBO_N_E_900_UNC
crystal_thickness_unc_mm = config.BBO_CRYSTAL_THICKNESS_UNC_MM

compensator_material = config.COMPENSATOR_MATERIAL
compensator_birefringence = config.COMPENSATOR_BIREFRINGENCE
compensator_birefringence_unc = config.COMPENSATOR_BIREFRINGENCE_UNC

# ============================================
# Helper functions
# ============================================

def residual_phase(phase: float) -> float:
    return phase % (2 * np.pi)


def compute_phase(delta_n: float, thickness_mm: float, wavelength_mm: float) -> float:
    return (2 * np.pi * delta_n * thickness_mm) / wavelength_mm


def compute_phase_uncertainty(
    delta_n: float,
    thickness_mm: float,
    wavelength_mm: float,
    n_o_unc: float,
    n_e_unc: float,
    thickness_unc_mm: float,
) -> float:
    dphi_dn = 2 * np.pi * thickness_mm / wavelength_mm
    dphi_dt = 2 * np.pi * delta_n / wavelength_mm
    return np.sqrt(
        (dphi_dn * n_o_unc) ** 2
        + (dphi_dn * n_e_unc) ** 2
        + (dphi_dt * thickness_unc_mm) ** 2
    )


def compute_displacement(phase_rad: float, wavelength_mm: float, birefringence: float) -> float:
    return (phase_rad * wavelength_mm) / (2 * np.pi * birefringence)


def compute_displacement_uncertainty(
    phase_rad: float,
    wavelength_mm: float,
    birefringence: float,
    phase_unc: float,
    birefringence_unc: float,
) -> float:
    dD_dphi = wavelength_mm / (2 * np.pi * birefringence)
    dD_db = -phase_rad * wavelength_mm / (2 * np.pi * birefringence ** 2)
    return np.sqrt((dD_dphi * phase_unc) ** 2 + (dD_db * birefringence_unc) ** 2)

# ============================================
# CALCULATIONS FOR SIGNAL (583 nm)
# ============================================

lambda_signal_nm = 583
lambda_signal_mm = lambda_signal_nm / 1_000_000

delta_n_signal = n_o_583 - n_e_583
phi_bbo_signal = compute_phase(delta_n_signal, crystal_thickness_mm, lambda_signal_mm)
phi_bbo_signal_unc = compute_phase_uncertainty(
    delta_n_signal,
    crystal_thickness_mm,
    lambda_signal_mm,
    n_o_583_unc,
    n_e_583_unc,
    crystal_thickness_unc_mm,
)

residual_signal = residual_phase(phi_bbo_signal)
compensator_phase_needed_signal = residual_phase(-residual_signal)

displacement_signal = compute_displacement(
    compensator_phase_needed_signal,
    lambda_signal_mm,
    compensator_birefringence,
)

displacement_signal_unc = compute_displacement_uncertainty(
    compensator_phase_needed_signal,
    lambda_signal_mm,
    compensator_birefringence,
    phi_bbo_signal_unc,
    compensator_birefringence_unc,
)

# ============================================
# CALCULATIONS FOR IDLER (900 nm)
# ============================================

lambda_idler_nm = 900
lambda_idler_mm = lambda_idler_nm / 1_000_000

delta_n_idler = n_o_900 - n_e_900
phi_bbo_idler = compute_phase(delta_n_idler, crystal_thickness_mm, lambda_idler_mm)
phi_bbo_idler_unc = compute_phase_uncertainty(
    delta_n_idler,
    crystal_thickness_mm,
    lambda_idler_mm,
    n_o_900_unc,
    n_e_900_unc,
    crystal_thickness_unc_mm,
)

residual_idler = residual_phase(phi_bbo_idler)
compensator_phase_needed_idler = residual_phase(-residual_idler)

displacement_idler = compute_displacement(
    compensator_phase_needed_idler,
    lambda_idler_mm,
    compensator_birefringence,
)

displacement_idler_unc = compute_displacement_uncertainty(
    compensator_phase_needed_idler,
    lambda_idler_mm,
    compensator_birefringence,
    phi_bbo_idler_unc,
    compensator_birefringence_unc,
)

# ============================================
# PRINT RESULTS
# ============================================

print("\n" + "=" * 70)
print("BABINET (SOLEIL-BABINET) COMPENSATOR CALCULATOR")
print("=" * 70)
print(f"\nBBO Crystal Thickness: {crystal_thickness_mm:.4f} ± {crystal_thickness_unc_mm:.4f} mm")
print(f"Compensator Material: {compensator_material} (Δn = {compensator_birefringence:.6f} ± {compensator_birefringence_unc:.6f})")

print("\n" + "-" * 50)
print("SIGNAL WAVELENGTH: 583 nm")
print("-" * 50)
print(f"  n_o = {n_o_583:.6f} ± {n_o_583_unc:.6f}")
print(f"  n_e = {n_e_583:.6f} ± {n_e_583_unc:.6f}")
print(f"  Δn (birefringence) = {delta_n_signal:.6f}")
print(f"  Phase from BBO = {phi_bbo_signal:.4f} ± {phi_bbo_signal_unc:.4f} rad ({np.degrees(phi_bbo_signal):.2f} ± {np.degrees(phi_bbo_signal_unc):.2f}°)")
print(f"  Residual phase (mod 2π) = {residual_signal:.4f} rad ({np.degrees(residual_signal):.2f}°)")
print(f"  Compensator needs: {compensator_phase_needed_signal:.4f} rad ({np.degrees(compensator_phase_needed_signal):.2f}°)")
print(f"  Compensator displacement: {displacement_signal * 1000:.3f} ± {displacement_signal_unc * 1000:.3f} μm")

print("\n" + "-" * 50)
print("IDLER WAVELENGTH: 900 nm")
print("-" * 50)
print(f"  n_o = {n_o_900:.6f} ± {n_o_900_unc:.6f}")
print(f"  n_e = {n_e_900:.6f} ± {n_e_900_unc:.6f}")
print(f"  Δn (birefringence) = {delta_n_idler:.6f}")
print(f"  Phase from BBO = {phi_bbo_idler:.4f} ± {phi_bbo_idler_unc:.4f} rad ({np.degrees(phi_bbo_idler):.2f} ± {np.degrees(phi_bbo_idler_unc):.2f}°)")
print(f"  Residual phase (mod 2π) = {residual_idler:.4f} rad ({np.degrees(residual_idler):.2f}°)")
print(f"  Compensator needs: {compensator_phase_needed_idler:.4f} rad ({np.degrees(compensator_phase_needed_idler):.2f}°)")
print(f"  Compensator displacement: {displacement_idler * 1000:.3f} ± {displacement_idler_unc * 1000:.3f} μm")

print("\n" + "=" * 70)
print("LABORATORY INSTRUCTIONS")
print("=" * 70)
print(f"1. Set the Babinet compensator to {np.degrees(compensator_phase_needed_signal):.1f}° retardation")
print(f"2. OR adjust micrometer to {displacement_signal * 1000:.1f} ± {displacement_signal_unc * 1000:.1f} μm displacement")
print("3. Fine-tune while monitoring coincidence counts")
print("4. Maximum coincidences = optimal compensation (phase = 0)")
print("=" * 70)

# ============================================
# SAVE RESULTS TO CSV
# ============================================

csv_path = config.RESULTS_CSV_DIR / "compensator_calculation.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["section", "parameter", "value", "units", "uncertainty", "uncertainty_units"])
    writer.writerow(["signal", "n_o", n_o_583, "", n_o_583_unc, ""])
    writer.writerow(["signal", "n_e", n_e_583, "", n_e_583_unc, ""])
    writer.writerow(["signal", "phase_from_bbo", phi_bbo_signal, "rad", phi_bbo_signal_unc, "rad"])
    writer.writerow(["signal", "compensator_phase_needed", compensator_phase_needed_signal, "rad", phi_bbo_signal_unc, "rad"])
    writer.writerow(["signal", "displacement_needed", displacement_signal * 1000, "μm", displacement_signal_unc * 1000, "μm"])
    writer.writerow([])
    writer.writerow(["idler", "n_o", n_o_900, "", n_o_900_unc, ""])
    writer.writerow(["idler", "n_e", n_e_900, "", n_e_900_unc, ""])
    writer.writerow(["idler", "phase_from_bbo", phi_bbo_idler, "rad", phi_bbo_idler_unc, "rad"])
    writer.writerow(["idler", "compensator_phase_needed", compensator_phase_needed_idler, "rad", phi_bbo_idler_unc, "rad"])
    writer.writerow(["idler", "displacement_needed", displacement_idler * 1000, "μm", displacement_idler_unc * 1000, "μm"])

print(f"\n✓ Calculation results saved to '{csv_path}'")

# ============================================
# CREATE CALIBRATION PLOT
# ============================================

plot_path = config.THESIS_FIG_DIR / "compensator_calibration.png"

displacements_um = np.linspace(0, 500, 1000)
phases_deg = (displacements_um / 1000) * (360 / (lambda_signal_mm / compensator_birefringence))

plt.figure(figsize=(10, 6))
plt.plot(displacements_um, phases_deg, "b-", linewidth=2)
plt.axhline(
    y=np.degrees(compensator_phase_needed_signal),
    color="r",
    linestyle="--",
    label=f"Target: {np.degrees(compensator_phase_needed_signal):.1f}°",
)
plt.axvline(
    x=displacement_signal * 1000,
    color="g",
    linestyle="--",
    label=f"Displacement: {displacement_signal * 1000:.1f} μm",
)

phase_unc_deg = np.degrees(phi_bbo_signal_unc)
target_disp_um = displacement_signal * 1000
target_disp_unc_um = displacement_signal_unc * 1000
plt.fill_between(
    displacements_um,
    np.degrees(compensator_phase_needed_signal) - phase_unc_deg,
    np.degrees(compensator_phase_needed_signal) + phase_unc_deg,
    color="red",
    alpha=0.08,
    label=f"Phase uncertainty ±{phase_unc_deg:.2f}°",
)
plt.axvspan(
    target_disp_um - target_disp_unc_um,
    target_disp_um + target_disp_unc_um,
    color="green",
    alpha=0.12,
    label=f"Displacement uncertainty ±{target_disp_unc_um:.1f} μm",
)

plt.xlabel("Compensator Displacement (micrometers)", fontsize=12)
plt.ylabel("Phase Retardation (degrees)", fontsize=12)
plt.title("Babinet Compensator Calibration Curve (583 nm)", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
print(f"\n✓ Calibration curve saved to '{plot_path}'")
plt.close()
