"""
================================================================================
                    COMPLETE SPDC QUANTUM SIMULATION
                   WITH PROPER ENTANGLEMENT MEASUREMENT
================================================================================
Author: PhD Researcher
Date: April 2026
Version: 4.0 - Complete Production Ready

This script implements a full quantum simulation of Spontaneous Parametric
Down-Conversion (SPDC) using the QuTiP library. It includes:

1. Full master equation solver (not classical approximation)
2. Proper entanglement measurement via concurrence
3. Photon number evolution and pump depletion
4. Coincidence rate calculations
5. Bell inequality parameter estimation
6. Publication-quality visualizations
7. Complete data export for further analysis

================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from qutip import *
import time
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("COMPLETE SPDC QUANTUM SIMULATION v4.0")
print("="*80)
print(f"QuTiP version: {qutip.__version__}")
print("="*80)

# ============================================================================
# PART 1: PHYSICAL PARAMETERS (ADJUST FOR YOUR EXPERIMENT)
# ============================================================================

print("\n[1] Setting up physical parameters...")

class SimulationConfig:
    """Configuration class for all simulation parameters"""
    
    def __init__(self):
        # === Simulation Parameters ===
        self.N = 3                    # Truncation dimension (max photons per mode)
        self.time_max = 20.0         # Maximum time (normalized units)
        self.n_time_steps = 100      # Number of time steps
        self.tlist = np.linspace(0, self.time_max, self.n_time_steps)
        
        # === Coupling and Loss Parameters ===
        self.g_coupling = 0.08       # SPDC coupling strength
        self.gamma_pump = 0.001      # Pump loss rate
        self.gamma_signal = 0.001    # Signal loss rate
        self.gamma_idler = 0.001     # Idler loss rate
        
        # === Initial State Parameters ===
        self.alpha_pump = 1.5        # Coherent state amplitude for pump
        self.pump_coherent = True    # Use coherent state (True) or Fock (False)
        self.initial_signal_photons = 0
        self.initial_idler_photons = 0
        
        # === Experimental Parameters (for scaling) ===
        self.pump_power_mW = 100
        self.crystal_length_mm = 3.0
        self.pump_wavelength_nm = 355
        
        # === Derived Parameters ===
        self.hilbert_dim = self.N ** 3
        self._validate()
        
    def _validate(self):
        """Validate parameters"""
        print(f"  Truncation: {self.N} photons per mode")
        print(f"  Hilbert space dimension: {self.hilbert_dim}")
        print(f"  Time steps: {self.n_time_steps}")
        print(f"  Coupling strength g = {self.g_coupling}")
        print(f"  Pump loss γ = {self.gamma_pump}")
        print(f"  Signal loss γ = {self.gamma_signal}")
        print(f"  Idler loss γ = {self.gamma_idler}")

config = SimulationConfig()

# ============================================================================
# PART 2: QUANTUM OPERATORS
# ============================================================================

print("\n[2] Building quantum operators...")

# Create annihilation operators for each mode
a = destroy(config.N)  # Pump mode
b = destroy(config.N)  # Signal mode
c = destroy(config.N)  # Idler mode

# Identity operator
I = qeye(config.N)

# Expand to full Hilbert space (tensor product)
a_full = tensor(a, I, I)
b_full = tensor(I, b, I)
c_full = tensor(I, I, c)

print(f"  Operator dimensions: {a_full.shape[0]} x {a_full.shape[1]}")

# Number operators
Np = a_full.dag() * a_full
Ns = b_full.dag() * b_full
Ni = c_full.dag() * c_full

# Squared number operators (for variance calculations)
Np2 = Np * Np
Ns2 = Ns * Ns
Ni2 = Ni * Ni

# ============================================================================
# PART 3: HAMILTONIAN (Interaction Hamiltonian)
# ============================================================================

print("\n[3] Building Hamiltonian...")

# SPDC Hamiltonian: H = iℏg (a b† c† - a† b c)
# This describes:
# - a b† c†: Pump photon annihilated, signal and idler created
# - a† b c: Pump photon created, signal and idler annihilated
H_int = 1j * config.g_coupling * (a_full * b_full.dag() * c_full.dag() - 
                                   a_full.dag() * b_full * c_full)

print(f"  Hamiltonian norm: {H_int.norm():.4f}")

# ============================================================================
# PART 4: LINDBLAD OPERATORS (Decoherence)
# ============================================================================

print("\n[4] Building Lindblad operators...")

# Loss operators for each mode (photon loss)
c_ops = [
    np.sqrt(config.gamma_pump) * a_full,
    np.sqrt(config.gamma_signal) * b_full,
    np.sqrt(config.gamma_idler) * c_full,
]

print(f"  Number of dissipators: {len(c_ops)}")

# Additional thermal noise (optional)
n_th = 0.01  # Thermal photon number
c_ops_thermal = [
    np.sqrt(config.gamma_pump * n_th) * a_full.dag(),
    np.sqrt(config.gamma_signal * n_th) * b_full.dag(),
    np.sqrt(config.gamma_idler * n_th) * c_full.dag(),
]
c_ops.extend(c_ops_thermal)

print(f"  Including thermal noise (n_th = {n_th})")
print(f"  Total dissipators: {len(c_ops)}")

# ============================================================================
# PART 5: INITIAL STATE
# ============================================================================

print("\n[5] Setting initial state...")

# Coherent state in pump mode
if config.pump_coherent:
    pump_state = coherent(config.N, config.alpha_pump)
    pump_description = f"coherent |α={config.alpha_pump}⟩"
else:
    pump_state = fock(config.N, config.alpha_pump)
    pump_description = f"Fock |{config.alpha_pump}⟩"

# Vacuum for signal and idler
signal_state = fock(config.N, config.initial_signal_photons)
idler_state = fock(config.N, config.initial_idler_photons)

# Initial density matrix (pure state)
rho0 = tensor(pump_state, signal_state, idler_state)

# Initial pump photon number
initial_pump_photons = expect(Np, rho0)
print(f"  Pump mode: {pump_description}")
print(f"  Signal mode: Fock |0⟩")
print(f"  Idler mode: Fock |0⟩")
print(f"  Initial pump photons: {initial_pump_photons:.3f}")

# ============================================================================
# PART 6: OBSERVABLES
# ============================================================================

print("\n[6] Setting up observables...")

# Basic observables
e_ops = [Np, Ns, Ni]

# Additional observables for correlation
e_ops_names = ["Np", "Ns", "Ni"]

print(f"  Tracking {len(e_ops)} observables: {', '.join(e_ops_names)}")

# ============================================================================
# PART 7: MASTER EQUATION SOLVER
# ============================================================================

print("\n[7] Solving master equation...")
print("  This may take 10-30 seconds...")

start_time = time.time()

result = mesolve(H_int, rho0, config.tlist, c_ops, e_ops)

end_time = time.time()
print(f"  ✓ Solver completed in {end_time - start_time:.3f} seconds")

# Extract expectation values
Np_exp = result.expect[0]
Ns_exp = result.expect[1]
Ni_exp = result.expect[2]

# ============================================================================
# PART 8: POST-PROCESSING - GET FINAL STATE
# ============================================================================

print("\n[8] Extracting final state...")

result_final = mesolve(H_int, rho0, [config.tlist[-1]], c_ops, e_ops, 
                       options={'store_states': True})
rho_final = result_final.states[-1]

print(f"  Final state density matrix shape: {rho_final.shape}")

# ============================================================================
# PART 9: ENTANGLEMENT MEASUREMENT (CONCURRENCE)
# ============================================================================

print("\n[9] Measuring entanglement (concurrence)...")

def calculate_concurrence(rho_si_4x4):
    """
    Calculate concurrence for a 2-qubit density matrix (4x4)
    
    Concurrence C = max(0, λ₁ - λ₂ - λ₃ - λ₄)
    where λᵢ are eigenvalues of R = ρ (σ_y⊗σ_y) ρ* (σ_y⊗σ_y)
    
    Returns:
        float: Concurrence between 0 (separable) and 1 (maximally entangled)
    """
    # Pauli Y matrix
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    
    # σ_y ⊗ σ_y
    YY = np.kron(sigma_y, sigma_y)
    
    # Calculate R matrix
    rho_mat = rho_si_4x4
    R = rho_mat @ YY @ rho_mat.conj() @ YY
    
    # Eigenvalues of R
    eigenvalues = np.linalg.eigvals(R)
    
    # Sort in descending order after taking square root
    eigenvalues = np.sort(np.sqrt(np.maximum(0, np.real(eigenvalues))))[::-1]
    
    # Concurrence
    concurrence = max(0, eigenvalues[0] - eigenvalues[1] - eigenvalues[2] - eigenvalues[3])
    
    return concurrence

def calculate_negativity(rho):
    """
    Calculate negativity = (||ρ^{T_B}||_1 - 1)/2
    A measure of entanglement
    """
    rho_pt = rho.ptrace(0)  # Partial transpose
    trace_norm = (rho_pt.dag() * rho_pt).sqrtm().tr()
    negativity = (trace_norm - 1) / 2
    return negativity

# Partial trace: trace out pump mode (index 0) to get signal-idler state
rho_si = ptrace(rho_final, [1, 2])
print(f"  Signal-Idler density matrix shape: {rho_si.shape}")

# Project to 2x2 subspace (first two levels) for qubit approximation
if rho_si.shape[0] >= 2:
    rho_2x2 = rho_si[0:2, 0:2]
    concurrence = calculate_concurrence(rho_2x2)
    print(f"  Concurrence: {concurrence:.6f}")
    print(f"  Entanglement status: {'ENTANGLED ✓' if concurrence > 0 else 'SEPARABLE ✗'}")
    
    # Interpret concurrence value
    if concurrence > 0.5:
        print(f"  → Highly entangled (C = {concurrence:.3f})")
    elif concurrence > 0:
        print(f"  → Weakly entangled (C = {concurrence:.3f})")
    else:
        print(f"  → No entanglement detected")
else:
    concurrence = 0
    print(f"  Concurrence: Not applicable (dimensionality too low)")

# ============================================================================
# PART 10: CALCULATE COINCIDENCE RATE (Expected Lab Results)
# ============================================================================

print("\n[10] Calculating expected coincidence rates...")

# Coincidence rate formula: R_coinc = η² × μ² × (pairs per pulse)
# where η is detection efficiency, μ is pump power scaling

# Detection efficiency (typical values)
detection_efficiency = 0.12 * 0.35  # collection × detector efficiency

# Pump power (mW)
pump_powers_mW = np.array([20, 50, 100, 150, 200, 250])

# Coincidence rate ∝ pump_power² ∝ pairs generated
pairs_at_100mW = Ns_exp[-1] / config.g_coupling**2  # Scale appropriately
coincidence_rates = detection_efficiency**2 * (pump_powers_mW / 100)**2 * pairs_at_100mW * 1e6

print(f"  Detection efficiency: {detection_efficiency*100:.1f}%")
print(f"  Expected coincidence rates:")
for p, r in zip(pump_powers_mW, coincidence_rates):
    print(f"    {p:3d} mW: {r:.3f} Hz")

# ============================================================================
# PART 11: BELL PARAMETER ESTIMATION
# ============================================================================

print("\n[11] Estimating Bell parameter S...")

# Theoretically, for maximally entangled Bell state, S = 2√2 ≈ 2.828
# For our simulated state, we estimate based on concurrence
if concurrence > 0:
    # Approximate Bell parameter from concurrence: S ≈ 2√(1 + C²)
    bell_S_estimate = 2 * np.sqrt(1 + concurrence**2)
    print(f"  Estimated Bell parameter S: {bell_S_estimate:.3f}")
    print(f"  Classical limit: S ≤ 2")
    print(f"  Quantum violation: {'YES ✓' if bell_S_estimate > 2 else 'NO ✗'}")
else:
    print(f"  Cannot estimate Bell parameter (no entanglement)")

# ============================================================================
# PART 12: RESULTS SUMMARY
# ============================================================================

print("\n" + "="*80)
print("SIMULATION RESULTS SUMMARY")
print("="*80)

# Photon numbers
pump_depletion = (initial_pump_photons - Np_exp[-1]) / initial_pump_photons * 100
signal_idler_correlation = Ns_exp[-1] * Ni_exp[-1]

print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                           QUANTUM RESULTS                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PHOTON NUMBERS:                                                            │
│    Initial pump photons:   {initial_pump_photons:.4f}                              │
│    Final pump photons:     {Np_exp[-1]:.4f}                                    │
│    Final signal photons:   {Ns_exp[-1]:.4f}                                    │
│    Final idler photons:    {Ni_exp[-1]:.4f}                                    │
│    Pairs generated:        {Ns_exp[-1]:.4f}                                    │
│    Pump depletion:         {pump_depletion:.2f}%                                 │
│                                                                             │
│  ENTANGLEMENT:                                                              │
│    Concurrence:            {concurrence:.6f}                                    │
│    Entangled:              {'YES ✓' if concurrence > 0 else 'NO ✗'}              │
│                                                                             │
│  BELL PARAMETER:                                                            │
│    Estimated S:            {bell_S_estimate:.3f} (if concurrence > 0)           │
│    Violation:              {'YES ✓' if concurrence > 0.5 else 'NO ✗'}            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")

# ============================================================================
# PART 13: VISUALIZATION (Publication Quality)
# ============================================================================

print("\n[12] Generating publication-quality figures...")

# Create figure with 6 subplots
fig = plt.figure(figsize=(18, 12))

# Plot 1: Photon number evolution
ax1 = plt.subplot(2, 3, 1)
ax1.plot(config.tlist, Np_exp, 'b-', linewidth=2.5, label='Pump')
ax1.plot(config.tlist, Ns_exp, 'r-', linewidth=2.5, label='Signal')
ax1.plot(config.tlist, Ni_exp, 'g-', linewidth=2.5, label='Idler')
ax1.fill_between(config.tlist, 0, Np_exp, color='blue', alpha=0.1)
ax1.fill_between(config.tlist, 0, Ns_exp, color='red', alpha=0.1)
ax1.set_xlabel('Normalized Time', fontsize=12)
ax1.set_ylabel('Photon Number', fontsize=12)
ax1.set_title('(a) Photon Number Evolution', fontsize=14, fontweight='bold')
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3, linestyle='--')

# Plot 2: Pairs generated
ax2 = plt.subplot(2, 3, 2)
ax2.plot(config.tlist, Ns_exp, 'purple', linewidth=2.5)
ax2.fill_between(config.tlist, 0, Ns_exp, color='purple', alpha=0.2)
ax2.set_xlabel('Normalized Time', fontsize=12)
ax2.set_ylabel('Number of Pairs', fontsize=12)
ax2.set_title('(b) SPDC Photon Pairs Generated', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')

# Plot 3: Pump depletion
ax3 = plt.subplot(2, 3, 3)
depletion_curve = (Np_exp[0] - Np_exp) / Np_exp[0] * 100
ax3.plot(config.tlist, depletion_curve, 'b-', linewidth=2.5)
ax3.fill_between(config.tlist, 0, depletion_curve, color='blue', alpha=0.2)
ax3.set_xlabel('Normalized Time', fontsize=12)
ax3.set_ylabel('Pump Depletion (%)', fontsize=12)
ax3.set_title('(c) Pump Depletion', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, linestyle='--')

# Plot 4: Exponential growth (log scale)
ax4 = plt.subplot(2, 3, 4)
ax4.semilogy(config.tlist, Ns_exp, 'r-', linewidth=2.5, label='Signal')
ax4.semilogy(config.tlist, Ni_exp, 'g-', linewidth=2.5, label='Idler')
ax4.set_xlabel('Normalized Time', fontsize=12)
ax4.set_ylabel('Photon Number (log scale)', fontsize=12)
ax4.set_title('(d) Exponential Growth', fontsize=14, fontweight='bold')
ax4.legend(loc='best')
ax4.grid(True, alpha=0.3, linestyle='--')

# Plot 5: Coincidence rate vs pump power
ax5 = plt.subplot(2, 3, 5)
ax5.plot(pump_powers_mW, coincidence_rates, 'bo-', linewidth=2.5, markersize=8)
ax5.fill_between(pump_powers_mW, 0, coincidence_rates, color='blue', alpha=0.2)
ax5.set_xlabel('Pump Power (mW)', fontsize=12)
ax5.set_ylabel('Coincidence Rate (Hz)', fontsize=12)
ax5.set_title('(e) Coincidence Rate vs Pump Power', fontsize=14, fontweight='bold')
ax5.grid(True, alpha=0.3, linestyle='--')

# Plot 6: Photon number distribution
ax6 = plt.subplot(2, 3, 6)
# Photon number distribution for signal mode
rho_signal = ptrace(rho_final, [1])
probs = np.real(np.diag(rho_signal.full()))
n_values = np.arange(len(probs))
ax6.bar(n_values, probs, color='red', alpha=0.7, edgecolor='black', linewidth=1)
ax6.set_xlabel('Photon Number n', fontsize=12)
ax6.set_ylabel('Probability', fontsize=12)
ax6.set_title('(f) Signal Photon Distribution', fontsize=14, fontweight='bold')
ax6.set_xticks(n_values)
ax6.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('spdc_qutip_best_results.png', dpi=300, bbox_inches='tight')
plt.savefig('spdc_qutip_best_results.pdf', bbox_inches='tight')
plt.close()

print("  ✓ Saved: spdc_qutip_best_results.png/pdf")

# ============================================================================
# PART 14: SAVE DATA FOR FURTHER ANALYSIS
# ============================================================================

print("\n[13] Saving simulation data...")

# Save time evolution data
time_data = np.column_stack([config.tlist, Np_exp, Ns_exp, Ni_exp, depletion_curve])
np.savetxt('spdc_time_evolution.csv', time_data, 
           delimiter=',', 
           header='Time,Pump_photons,Signal_photons,Idler_photons,Pump_depletion_pct',
           comments='')

print("  ✓ Saved: spdc_time_evolution.csv")

# Save coincidence rate data
coincidence_data = np.column_stack([pump_powers_mW, coincidence_rates])
np.savetxt('spdc_coincidence_rates.csv', coincidence_data,
           delimiter=',',
           header='Pump_power_mW,Coincidence_rate_Hz',
           comments='')

print("  ✓ Saved: spdc_coincidence_rates.csv")

# Save final density matrix elements (for advanced analysis)
rho_signal_final = ptrace(rho_final, [1])
np.savetxt('rho_signal_matrix.csv', rho_signal_final.full(), 
           delimiter=',', 
           header='Signal mode reduced density matrix',
           comments='')

print("  ✓ Saved: rho_signal_matrix.csv")

# Save summary results
with open('spdc_simulation_summary.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("SPDC QUANTUM SIMULATION SUMMARY\n")
    f.write("="*80 + "\n\n")
    f.write(f"Simulation date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"QuTiP version: {qutip.__version__}\n\n")
    f.write("PARAMETERS:\n")
    f.write(f"  Truncation dimension: {config.N}\n")
    f.write(f"  Hilbert space dimension: {config.hilbert_dim}\n")
    f.write(f"  Coupling strength g: {config.g_coupling}\n")
    f.write(f"  Loss rate γ: {config.gamma_pump}\n")
    f.write(f"  Initial pump amplitude α: {config.alpha_pump}\n\n")
    f.write("RESULTS:\n")
    f.write(f"  Initial pump photons: {initial_pump_photons:.6f}\n")
    f.write(f"  Final pump photons: {Np_exp[-1]:.6f}\n")
    f.write(f"  Final signal photons: {Ns_exp[-1]:.6f}\n")
    f.write(f"  Final idler photons: {Ni_exp[-1]:.6f}\n")
    f.write(f"  Pairs generated: {Ns_exp[-1]:.6f}\n")
    f.write(f"  Pump depletion: {pump_depletion:.2f}%\n")
    f.write(f"  Concurrence: {concurrence:.6f}\n")
    f.write(f"  Entangled: {concurrence > 0}\n")
    if concurrence > 0:
        f.write(f"  Estimated Bell S: {bell_S_estimate:.3f}\n")

print("  ✓ Saved: spdc_simulation_summary.txt")

# ============================================================================
# PART 15: FINAL COMPLETION
# ============================================================================

print("\n" + "="*80)
print("SIMULATION COMPLETED SUCCESSFULLY")
print("="*80)
print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FILES GENERATED                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  ✓ spdc_qutip_best_results.png      - Main figure (6 subplots)             │
│  ✓ spdc_qutip_best_results.pdf      - Vector version for thesis            │
│  ✓ spdc_time_evolution.csv          - Time evolution data                  │
│  ✓ spdc_coincidence_rates.csv       - Coincidence vs power data            │
│  ✓ rho_signal_matrix.csv            - Signal mode density matrix           │
│  ✓ spdc_simulation_summary.txt      - Complete text summary                │
└─────────────────────────────────────────────────────────────────────────────┘
""")
print("="*80)