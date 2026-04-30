"""
================================================================================
COMPLETE SPDC QUANTUM SIMULATION WITH ENTANGLEMENT MEASUREMENT
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from qutip import *
import time

print("="*80)
print("SPDC SIMULATION WITH ENTANGLEMENT MEASUREMENT")
print("="*80)
print(f"QuTiP version: {qutip.__version__}")
print("="*80)

# ============================================================================
# PART 1: SIMULATION PARAMETERS
# ============================================================================

N = 3  # Truncation dimension (3x3x3 = 27 states)
tlist = np.linspace(0, 20, 80)
g = 0.08  # Coupling strength
gamma = 0.001  # Loss rate

print("\nParameters:")
print(f"  Truncation: {N} (max {N} photons)")
print(f"  Hilbert space: {N**3}")
print(f"  Coupling g = {g}")
print(f"  Loss γ = {gamma}")

# ============================================================================
# PART 2: CREATE OPERATORS
# ============================================================================

a = destroy(N)
b = destroy(N)
c = destroy(N)
I = qeye(N)

a_full = tensor(a, I, I)
b_full = tensor(I, b, I)
c_full = tensor(I, I, c)

Np = a_full.dag() * a_full
Ns = b_full.dag() * b_full
Ni = c_full.dag() * c_full

# ============================================================================
# PART 3: HAMILTONIAN
# ============================================================================

H = 1j * g * (a_full * b_full.dag() * c_full.dag() - a_full.dag() * b_full * c_full)

# ============================================================================
# PART 4: LINDBLAD OPERATORS
# ============================================================================

c_ops = [
    np.sqrt(gamma) * a_full,
    np.sqrt(gamma) * b_full,
    np.sqrt(gamma) * c_full,
]

# ============================================================================
# PART 5: INITIAL STATE
# ============================================================================

alpha = 1.5
rho0 = tensor(coherent(N, alpha), fock(N, 0), fock(N, 0))

print(f"Initial pump photons: {expect(Np, rho0):.2f}")

# ============================================================================
# PART 6: RUN SIMULATION
# ============================================================================

print("\nSolving master equation...")
start = time.time()

result = mesolve(H, rho0, tlist, c_ops, [Np, Ns, Ni])

print(f"Solved in {time.time()-start:.2f}s")

Np_exp = result.expect[0]
Ns_exp = result.expect[1]
Ni_exp = result.expect[2]

# ============================================================================
# PART 7: GET FINAL STATE
# ============================================================================

result_final = mesolve(H, rho0, [tlist[-1]], c_ops, [Np, Ns, Ni], 
                       options={'store_states': True})
rho_final = result_final.states[-1]

# ============================================================================
# PART 8: CALCULATE CONCURRENCE
# ============================================================================

print("\nCalculating concurrence...")

# Partial trace to get signal-idler
rho_si = ptrace(rho_final, [1, 2])

# Take first 2 levels (qubit approximation)
rho_2x2 = rho_si[0:2, 0:2]

def concurrence_2x2(rho):
    """Calculate concurrence for 2x2 density matrix"""
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    YY = np.kron(sigma_y, sigma_y)
    
    R = rho.full() @ YY @ rho.full().conj() @ YY
    eigvals = np.linalg.eigvals(R)
    eigvals = np.sort(np.sqrt(np.maximum(0, np.real(eigvals))))[::-1]
    
    return max(0, eigvals[0] - eigvals[1] - eigvals[2] - eigvals[3])

conc = concurrence_2x2(rho_2x2)
print(f"Concurrence: {conc:.6f}")
print(f"State is {'ENTANGLED' if conc > 0 else 'SEPARABLE'}")

# ============================================================================
# PART 9: RESULTS
# ============================================================================

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"""
Initial pump: {expect(Np, rho0):.2f}
Final pump: {Np_exp[-1]:.4f}
Final signal: {Ns_exp[-1]:.4f}
Final idler: {Ni_exp[-1]:.4f}
Pairs: {Ns_exp[-1]:.4f}
Depletion: {(Np_exp[0]-Np_exp[-1])/Np_exp[0]*100:.1f}%
Concurrence: {conc:.6f}
""")

# ============================================================================
# PART 10: PLOTS
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot(tlist, Np_exp, 'b-', label='Pump')
axes[0, 0].plot(tlist, Ns_exp, 'r-', label='Signal')
axes[0, 0].plot(tlist, Ni_exp, 'g-', label='Idler')
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Photon Number')
axes[0, 0].set_title('Photon Number Evolution')
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(tlist, Ns_exp, 'purple', linewidth=2)
axes[0, 1].set_xlabel('Time')
axes[0, 1].set_ylabel('Pairs')
axes[0, 1].set_title('Pairs Generated')
axes[0, 1].grid(True)

depletion = (Np_exp[0] - Np_exp) / Np_exp[0] * 100
axes[1, 0].plot(tlist, depletion, 'b-', linewidth=2)
axes[1, 0].set_xlabel('Time')
axes[1, 0].set_ylabel('Depletion (%)')
axes[1, 0].set_title('Pump Depletion')
axes[1, 0].grid(True)

axes[1, 1].semilogy(tlist, Ns_exp, 'r-', label='Signal')
axes[1, 1].semilogy(tlist, Ni_exp, 'g-', label='Idler')
axes[1, 1].set_xlabel('Time')
axes[1, 1].set_ylabel('Photon Number')
axes[1, 1].set_title('Exponential Growth')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('spdc_qutip_corrected.png', dpi=300)
plt.savefig('spdc_qutip_corrected.pdf')
plt.close()

print("Saved: spdc_qutip_corrected.png/pdf")

# Save data
data = np.column_stack([tlist, Np_exp, Ns_exp, Ni_exp, depletion])
np.savetxt('qutip_simulation_data.csv', data, delimiter=',', 
           header='Time,Pump,Signal,Idler,Depletion')

print("\nDONE")