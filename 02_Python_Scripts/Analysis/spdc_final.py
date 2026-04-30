"""
================================================================================
                    COMPLETE SPDC QUANTUM SIMULATION - FINAL
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from qutip import *
import time

print("="*80)
print("SPDC QUANTUM SIMULATION - FINAL VERSION")
print("="*80)
print(f"QuTiP version: {qutip.__version__}")
print("="*80)

# ============================================================================
# PARAMETERS
# ============================================================================

N = 4                          # Increase from 3 to 4
tlist = np.linspace(0, 20, 120)
g = 0.15                       # Increase coupling
gamma = 0.002                  # Reduce loss
alpha = 2.0                    # Increase pump amplitude

print(f"\nParameters:")
print(f"  Truncation: {N} (max {N} photons)")
print(f"  Hilbert space: {N**3}")
print(f"  Coupling g = {g}")
print(f"  Loss γ = {gamma}")
print(f"  Pump amplitude α = {alpha}")

# ============================================================================
# OPERATORS
# ============================================================================

print("\nCreating operators...")

a = destroy(N)   # Pump
b = destroy(N)   # Signal
c = destroy(N)   # Idler
I = qeye(N)

a_full = tensor(a, I, I)
b_full = tensor(I, b, I)
c_full = tensor(I, I, c)

Np = a_full.dag() * a_full
Ns = b_full.dag() * b_full
Ni = c_full.dag() * c_full

# ============================================================================
# HAMILTONIAN
# ============================================================================

print("Building Hamiltonian...")

H = 1j * g * (a_full * b_full.dag() * c_full.dag() - 
              a_full.dag() * b_full * c_full)

print(f"  Hamiltonian norm: {H.norm():.4f}")

# ============================================================================
# LINDBLAD OPERATORS
# ============================================================================

print("Building Lindblad operators...")

c_ops = [
    np.sqrt(gamma) * a_full,
    np.sqrt(gamma) * b_full,
    np.sqrt(gamma) * c_full,
]

# ============================================================================
# INITIAL STATE
# ============================================================================

print("Setting initial state...")

rho0 = tensor(coherent(N, alpha), fock(N, 0), fock(N, 0))
initial_pump = expect(Np, rho0)
print(f"  Initial pump photons: {initial_pump:.3f}")

# ============================================================================
# RUN SIMULATION
# ============================================================================

print("\nSolving master equation...")
start = time.time()

result = mesolve(H, rho0, tlist, c_ops, [Np, Ns, Ni])

print(f"Solved in {time.time()-start:.3f}s")

Np_exp = result.expect[0]
Ns_exp = result.expect[1]
Ni_exp = result.expect[2]

# ============================================================================
# FINAL STATE
# ============================================================================

print("\nExtracting final state...")

result_final = mesolve(H, rho0, [tlist[-1]], c_ops, [Np, Ns, Ni], 
                       options={'store_states': True})
rho_final = result_final.states[-1]

# ============================================================================
# CONCURRENCE CALCULATION (FIXED)
# ============================================================================

print("\nCalculating concurrence...")

# Partial trace to get signal-idler state
rho_si = ptrace(rho_final, [1, 2])

# Project to 2x2 subspace for qubit entanglement
# Take only the first 2 levels of signal and idler
dim_s = rho_si.shape[0]
if dim_s >= 4:
    # Extract 2x2x2x2 = 4x4 subspace
    indices = [0, 1]  # First two levels
    rho_4x4 = np.zeros((4, 4), dtype=complex)
    
    # Map indices
    for i1, idx1 in enumerate(indices):
        for j1, idxj1 in enumerate(indices):
            for i2, idx2 in enumerate(indices):
                for j2, idxj2 in enumerate(indices):
                    row = i1 * 2 + i2
                    col = j1 * 2 + j2
                    orig_row = idx1 * N + idx2
                    orig_col = idxj1 * N + idxj2
                    rho_4x4[row, col] = rho_si[orig_row, orig_col]
    
    def concurrence_2qubit(rho_4x4):
        """Calculate concurrence for 2-qubit state (4x4 matrix)"""
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        YY = np.kron(sigma_y, sigma_y)
        
        R = rho_4x4 @ YY @ rho_4x4.conj() @ YY
        eigvals = np.linalg.eigvals(R)
        eigvals = np.sort(np.sqrt(np.maximum(0, np.real(eigvals))))[::-1]
        
        return max(0, eigvals[0] - eigvals[1] - eigvals[2] - eigvals[3])
    
    concurrence = concurrence_2qubit(rho_4x4)
    print(f"  Concurrence: {concurrence:.6f}")
    
    if concurrence > 0.5:
        print(f"  → Highly entangled")
    elif concurrence > 0:
        print(f"  → Weakly entangled")
    else:
        print(f"  → Not entangled")
else:
    concurrence = 0
    print(f"  Concurrence: Cannot compute (dim={dim_s})")

# ============================================================================
# RESULTS
# ============================================================================

print("\n" + "="*60)
print("RESULTS")
print("="*60)

depletion = (initial_pump - Np_exp[-1]) / initial_pump * 100

print(f"""
Initial pump:     {initial_pump:.3f}
Final pump:       {Np_exp[-1]:.4f}
Final signal:     {Ns_exp[-1]:.4f}
Final idler:      {Ni_exp[-1]:.4f}
Pairs generated:  {Ns_exp[-1]:.4f}
Pump depletion:   {depletion:.1f}%
Concurrence:      {concurrence:.6f}
""")

# ============================================================================
# PLOTS
# ============================================================================

print("\nGenerating plots...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Photon numbers
axes[0, 0].plot(tlist, Np_exp, 'b-', linewidth=2, label='Pump')
axes[0, 0].plot(tlist, Ns_exp, 'r-', linewidth=2, label='Signal')
axes[0, 0].plot(tlist, Ni_exp, 'g-', linewidth=2, label='Idler')
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Photon Number')
axes[0, 0].set_title('(a) Photon Number Evolution')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Pairs
axes[0, 1].plot(tlist, Ns_exp, 'purple', linewidth=2)
axes[0, 1].set_xlabel('Time')
axes[0, 1].set_ylabel('Pairs')
axes[0, 1].set_title('(b) Pairs Generated')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Depletion
dep_curve = (Np_exp[0] - Np_exp) / Np_exp[0] * 100
axes[0, 2].plot(tlist, dep_curve, 'b-', linewidth=2)
axes[0, 2].set_xlabel('Time')
axes[0, 2].set_ylabel('Depletion (%)')
axes[0, 2].set_title('(c) Pump Depletion')
axes[0, 2].grid(True, alpha=0.3)

# Plot 4: Log scale
axes[1, 0].semilogy(tlist, Ns_exp, 'r-', linewidth=2, label='Signal')
axes[1, 0].semilogy(tlist, Ni_exp, 'g-', linewidth=2, label='Idler')
axes[1, 0].set_xlabel('Time')
axes[1, 0].set_ylabel('Photon Number (log)')
axes[1, 0].set_title('(d) Exponential Growth')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 5: Power scaling
powers = np.array([20, 50, 100, 150, 200, 250])
rates = (powers/100)**2 * Ns_exp[-1] * 1e6
axes[1, 1].plot(powers, rates, 'bo-', linewidth=2, markersize=6)
axes[1, 1].set_xlabel('Pump Power (mW)')
axes[1, 1].set_ylabel('Coincidence Rate (Hz)')
axes[1, 1].set_title('(e) Rate vs Pump Power')
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: Photon distribution
rho_s = ptrace(rho_final, [1])
probs = np.real(np.diag(rho_s.full()))
n_vals = np.arange(len(probs))
axes[1, 2].bar(n_vals, probs, color='red', alpha=0.7, edgecolor='black')
axes[1, 2].set_xlabel('Photon Number n')
axes[1, 2].set_ylabel('Probability')
axes[1, 2].set_title('(f) Signal Photon Distribution')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('spdc_final_results.png', dpi=300, bbox_inches='tight')
plt.savefig('spdc_final_results.pdf', bbox_inches='tight')
plt.close()

print("✓ Saved: spdc_final_results.png/pdf")

# ============================================================================
# SAVE DATA
# ============================================================================

data = np.column_stack([tlist, Np_exp, Ns_exp, Ni_exp, dep_curve])
np.savetxt('spdc_final_data.csv', data, delimiter=',',
           header='Time,Pump,Signal,Idler,Depletion_pct')

print("✓ Saved: spdc_final_data.csv")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("SIMULATION COMPLETE")
print("="*80)
print(f"""
FILES CREATED:
--------------
1. spdc_final_results.png/pdf - 6-panel figure
2. spdc_final_data.csv - Raw simulation data

KEY RESULTS:
-----------
Pairs generated: {Ns_exp[-1]:.4f}
Pump depletion: {depletion:.1f}%
Concurrence: {concurrence:.6f}
""")
print("="*80)