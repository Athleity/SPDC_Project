

import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("PhD-LEVEL: ENTANGLEMENT WITNESS - CORRECTED")
print("="*70)

def bell_state_phi_plus():
    rho = np.zeros((4, 4), dtype=complex)
    rho[0, 0] = 0.5
    rho[3, 3] = 0.5
    rho[0, 3] = 0.5
    rho[3, 0] = 0.5
    return rho

def noisy_state(rho_ideal, fidelity):
    mixed = np.eye(4, dtype=complex) / 4
    return fidelity * rho_ideal + (1 - fidelity) * mixed

def entanglement_witness():
    """W = I/4 - |Φ⁺⟩⟨Φ⁺|"""
    I4 = np.eye(4, dtype=complex) / 4
    phi = bell_state_phi_plus()
    return I4 - phi

W = entanglement_witness()
ideal_state = bell_state_phi_plus()
tr_W_ideal = np.real(np.trace(W @ ideal_state))

print(f"\nEntanglement witness W = I/4 - |Φ⁺⟩⟨Φ⁺|")
print(f"  Tr(W|Φ⁺⟩⟨Φ⁺|) = {tr_W_ideal:.4f}")
print(f"  {'✅ CORRECT (negative)' if tr_W_ideal < 0 else '❌ INCORRECT'}")

fidelities = np.linspace(0, 1, 21)
witness_values = []

print("\nFidelity → Witness value:")
for f in fidelities:
    rho = noisy_state(ideal_state, f)
    val = np.real(np.trace(W @ rho))
    witness_values.append(val)
    entangled = "✓ ENTANGLED" if val < 0 else "✗ Separable"
    print(f"  F = {f:.3f} → Tr(Wρ) = {val:.4f} ({entangled})")

exp_fidelity = 0.9622
exp_state = noisy_state(ideal_state, exp_fidelity)
exp_witness = np.real(np.trace(W @ exp_state))

print("\n" + "="*70)
print("EXPERIMENTAL PREDICTION")
print("="*70)
print(f"Your state fidelity: {exp_fidelity:.4f}")
print(f"Tr(Wρ) = {exp_witness:.4f}")
print(f"Entanglement: {'YES ✓' if exp_witness < 0 else 'NO ✗'}")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(fidelities, witness_values, 'b-', linewidth=2)
ax.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Entanglement boundary')
ax.axvline(x=exp_fidelity, color='purple', linestyle='--', label=f'Your fidelity: {exp_fidelity:.3f}')
ax.plot(exp_fidelity, exp_witness, 'ro', markersize=10, label='Your prediction')
ax.fill_between(fidelities, -0.5, 0, where=np.array(witness_values) < 0, color='green', alpha=0.2, label='Entangled region')
ax.fill_between(fidelities, 0, 0.5, where=np.array(witness_values) >= 0, color='red', alpha=0.2, label='Separable region')
ax.set_xlabel('Fidelity')
ax.set_ylabel('Tr(Wρ)')
ax.set_title('Entanglement Witness')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.3, 0.3)

plt.tight_layout()
plt.savefig('entanglement_witness_result.png', dpi=300, bbox_inches='tight')
plt.savefig('entanglement_witness_result.pdf', bbox_inches='tight')
plt.close()

print("\n✓ Saved: entanglement_witness_result.png/pdf")
print("="*70)