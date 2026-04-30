"""
================================================================================
                FINAL SPDC SIMULATION - STRONG COUPLING
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from qutip import *
import time

print("="*80)
print("FINAL SPDC SIMULATION - STRONG COUPLING REGIME")
print("="*80)
print(f"QuTiP version: {qutip.__version__}")
print("="*80)

# ============================================================================
# PARAMETERS (INCREASED FOR ENTANGLEMENT)
# ============================================================================

N = 5                     # Photon number truncation
tlist = np.linspace(0, 20, 100)
g = 0.25                  # Increased coupling strength
gamma = 0.003             # Loss rate
alpha = 3.0               # Strong pump

print(f"\nParameters:")
print(f"  Truncation: {N} (max {N} photons)")
print(f"  Hilbert space dim: {N**3}")
print(f"  Coupling g = {g}")
print(f"  Loss γ = {gamma}")
print(f"  Pump amplitude α = {alpha}")

# ============================================================================
# OPERATORS
# ============================================================================

print("\nCreating operators...")

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
# HAMILTONIAN & LINDBLAD
# ============================================================================

print("Building Hamiltonian...")
H = 1j * g * (a_full * b_full.dag() * c_full.dag() - a_full.dag() * b_full * c_full)

c_ops = [np.sqrt(gamma) * a_full,
         np.sqrt(gamma) * b_full,
         np.sqrt(gamma) * c_full]

# ============================================================================
# INITIAL STATE
# ============================================================================

print("Setting initial state...")
rho0 = tensor(coherent(N, alpha), fock(N, 0), fock(N, 0))
initial_pump = expect(Np, rho0)
print(f"  Initial pump photons: {initial_pump:.3f}")

# ============================================================================
# TIME EVOLUTION
# ============================================================================

print("\nSolving master equation...")
start = time.time()
result = mesolve(H, rho0, tlist, c_ops, [Np, Ns, Ni])
print(f"Solved in {time.time()-start:.3f}s")

Np_exp = result.expect[0]
Ns_exp = result.expect[1]
Ni_exp = result.expect[2]

# ============================================================================
# FINAL STATE AND ENTANGLEMENT
# ============================================================================

print("\nExtracting final state...")
result_final = mesolve(H, rho0, [tlist[-1]], c_ops, [Np, Ns, Ni],
                       options={'store_states': True})
rho_final = result_final.states[-1]

# Reduce to signal-idler subspace
rho_si = ptrace(rho_final, [1, 2])
rho_si.dims = [[N, N], [N, N]]

# Manual partial transpose on idler (second subsystem)
dim = N
total_dim = dim * dim
rho_si_np = rho_si.full()
rho_pt = np.zeros((total_dim, total_dim), dtype=complex)

# Map indices: row = i*dim + j, col = ip*dim + jp -> new row = i*dim + jp, new col = ip*dim + j
for i in range(dim):
    for j in range(dim):
        for ip in range(dim):
            for jp in range(dim):
                rho_pt[i*dim + jp, ip*dim + j] = rho_si_np[i*dim + j, ip*dim + jp]

# Compute eigenvalues of the partial transpose
rho_pt_qobj = Qobj(rho_pt, dims=[[N, N], [N, N]])
eigvals = rho_pt_qobj.eigenenergies()
neg_evals = eigvals[eigvals < -1e-8]

# Trace norm and logarithmic negativity
trace_norm = np.sum(np.linalg.svd(rho_pt)[1])
log_neg = np.log2(trace_norm) if trace_norm > 0 else 0
negativity = (trace_norm - 1) / 2

print(f"\nPartial transpose eigenvalues (first 5): {eigvals[:5]}")
print(f"Negative eigenvalues: {neg_evals}")
print(f"Trace norm: {trace_norm:.6f}")
print(f"Logarithmic negativity: {log_neg:.6f}")
print(f"Negativity: {negativity:.6f}")

if log_neg > 1e-6:
    print("  → State is ENTANGLED (log negativity > 0)")
else:
    print("  → State appears SEPARABLE (increase coupling further)")

# Additional measures
purity_si = (rho_si * rho_si).tr()
entropy_si = entropy_vn(rho_si)

# ============================================================================
# RESULTS SUMMARY
# ============================================================================

depletion = (initial_pump - Np_exp[-1]) / initial_pump * 100

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"""
Initial pump photons:  {initial_pump:.3f}
Final pump photons:    {Np_exp[-1]:.4f}
Final signal photons:  {Ns_exp[-1]:.4f}
Final idler photons:   {Ni_exp[-1]:.4f}
Pairs generated:       {Ns_exp[-1]:.4f}
Pump depletion:        {depletion:.1f}%

Entanglement:
  Negative eigenvalues: {len(neg_evals)}
  Log negativity:       {log_neg:.6f}
  Purity (ρ_si):        {purity_si:.6f}
  Entropy (ρ_si):       {entropy_si:.6f} nats
""")

# ============================================================================
# PLOTS
# ============================================================================

print("\nGenerating plots...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Photon numbers
axes[0, 0].plot(tlist, Np_exp, 'b-', label='Pump')
axes[0, 0].plot(tlist, Ns_exp, 'r-', label='Signal')
axes[0, 0].plot(tlist, Ni_exp, 'g-', label='Idler')
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Photon Number')
axes[0, 0].set_title('(a) Photon Evolution')
axes[0, 0].legend()
axes[0, 0].grid(True)

# 2. Pairs generated
axes[0, 1].plot(tlist, Ns_exp, 'purple', linewidth=2)
axes[0, 1].set_xlabel('Time')
axes[0, 1].set_ylabel('Pairs')
axes[0, 1].set_title('(b) Pairs Generated')
axes[0, 1].grid(True)

# 3. Pump depletion
dep_curve = (Np_exp[0] - Np_exp) / Np_exp[0] * 100
axes[0, 2].plot(tlist, dep_curve, 'b-', linewidth=2)
axes[0, 2].set_xlabel('Time')
axes[0, 2].set_ylabel('Depletion (%)')
axes[0, 2].set_title('(c) Pump Depletion')
axes[0, 2].grid(True)

# 4. Exponential growth
axes[1, 0].semilogy(tlist, Ns_exp, 'r-', label='Signal')
axes[1, 0].semilogy(tlist, Ni_exp, 'g-', label='Idler')
axes[1, 0].set_xlabel('Time')
axes[1, 0].set_ylabel('Photon Number (log)')
axes[1, 0].set_title('(d) Exponential Growth')
axes[1, 0].legend()
axes[1, 0].grid(True)

# 5. Entanglement measures
axes[1, 1].text(0.5, 0.5, f'Log Negativity = {log_neg:.4f}\nNegativity = {negativity:.4f}\nEntropy = {entropy_si:.4f} nats\nPurity = {purity_si:.4f}',
                ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
axes[1, 1].set_title('(e) Entanglement Measures')
axes[1, 1].axis('off')

# 6. Signal photon distribution
rho_s = ptrace(rho_final, [1])
probs = np.real(np.diag(rho_s.full()))
n_vals = np.arange(len(probs))
axes[1, 2].bar(n_vals, probs, color='red', alpha=0.7, edgecolor='black')
axes[1, 2].set_xlabel('Photon Number n')
axes[1, 2].set_ylabel('Probability')
axes[1, 2].set_title('(f) Signal Photon Distribution')
axes[1, 2].grid(True)

plt.tight_layout()
plt.savefig('spdc_final_entanglement.png', dpi=300)
plt.savefig('spdc_final_entanglement.pdf')
plt.close()

print("✓ Saved: spdc_final_entanglement.png/pdf")

# Save data
data = np.column_stack([tlist, Np_exp, Ns_exp, Ni_exp, dep_curve])
np.savetxt('spdc_final_data.csv', data, delimiter=',',
           header='Time,Pump,Signal,Idler,Depletion_pct')

print("\n" + "="*80)
print("SIMULATION COMPLETE")
print("="*80)