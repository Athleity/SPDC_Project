

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import sqrtm

print("="*70)
print("PhD-LEVEL: QUANTUM STATE TOMOGRAPHY")
print("="*70)

# ================================================================
# BELL STATE
# ================================================================

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

# ================================================================
# PROJECTORS
# ================================================================

def create_projectors():
    H = np.array([1, 0], dtype=complex)
    V = np.array([0, 1], dtype=complex)
    D = np.array([1, 1], dtype=complex) / np.sqrt(2)
    A = np.array([1, -1], dtype=complex) / np.sqrt(2)
    R = np.array([1, -1j], dtype=complex) / np.sqrt(2)
    L = np.array([1, 1j], dtype=complex) / np.sqrt(2)
    
    states = {'H': H, 'V': V, 'D': D, 'A': A, 'R': R, 'L': L}
    
    projectors = {}
    for name1, vec1 in states.items():
        proj1 = np.outer(vec1, vec1.conj())
        for name2, vec2 in states.items():
            proj2 = np.outer(vec2, vec2.conj())
            key = f"{name1}{name2}"
            projectors[key] = np.kron(proj1, proj2)
    return projectors

# ================================================================
# SIMULATE MEASUREMENTS
# ================================================================

def simulate_measurements(rho, projectors, total_counts=10000):
    measurements = {}
    for key, proj in projectors.items():
        prob = np.real(np.trace(rho @ proj))
        prob = max(0, min(1, prob))
        measurements[key] = np.random.poisson(prob * total_counts)
    return measurements

# ================================================================
# RECONSTRUCTION
# ================================================================

def reconstruct_density_matrix(measurements, projectors):
    total = sum(measurements.values())
    exp_probs = {k: v/total for k, v in measurements.items()}
    
    def rho_from_T(T):
        T_mat = T.reshape(4, 4)
        rho = T_mat.conj().T @ T_mat
        return rho / np.trace(rho)
    
    def neg_log_likelihood(T):
        rho = rho_from_T(T)
        log_lik = 0
        for key, proj in projectors.items():
            pred = np.real(np.trace(rho @ proj))
            pred = max(1e-10, min(1, pred))
            if exp_probs[key] > 0:
                log_lik += exp_probs[key] * np.log(pred)
        return -log_lik
    
    T0 = np.eye(4).flatten() * 0.5
    result = minimize(neg_log_likelihood, T0, method='L-BFGS-B', options={'maxiter': 1000})
    rho = rho_from_T(result.x)
    rho = (rho + rho.conj().T) / 2
    rho = rho / np.trace(rho)
    return rho

# ================================================================
# METRICS
# ================================================================

def calculate_fidelity(rho, target):
    sqrt_target = sqrtm(target)
    sqrt_target_rho = sqrt_target @ rho @ sqrt_target
    evals = np.linalg.eigvals(sqrt_target_rho)
    return (np.sum(np.sqrt(np.maximum(0, evals))))**2

def calculate_concurrence(rho):
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_y_y = np.kron(sigma_y, sigma_y)
    R = rho @ sigma_y_y @ rho.conj() @ sigma_y_y
    evals = np.linalg.eigvals(R)
    evals = np.sort(np.sqrt(np.abs(evals)))[::-1]
    return max(0, evals[0] - evals[1] - evals[2] - evals[3])

def calculate_bell_parameter(rho):
    def corr(rho, a, b):
        a_r, b_r = np.radians(a), np.radians(b)
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        A = np.cos(2*a_r) * sigma_z + np.sin(2*a_r) * sigma_x
        B = np.cos(2*b_r) * sigma_z + np.sin(2*b_r) * sigma_x
        return np.real(np.trace(rho @ np.kron(A, B)))
    
    a, ap, b, bp = 0, 45, 22.5, 67.5
    E1 = corr(rho, a, b)
    E2 = corr(rho, a, bp)
    E3 = corr(rho, ap, b)
    E4 = corr(rho, ap, bp)
    return abs(E1 - E2 + E3 + E4)

# ================================================================
# PLOT
# ================================================================

def plot_density_matrix(rho, title="Reconstructed State"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    labels = ['HH', 'HV', 'VH', 'VV']
    
    real_part = np.real(rho)
    im1 = ax1.imshow(real_part, cmap='RdBu', vmin=-0.5, vmax=0.5)
    ax1.set_title(f'{title}\nReal Part')
    ax1.set_xticks(range(4))
    ax1.set_yticks(range(4))
    ax1.set_xticklabels(labels)
    ax1.set_yticklabels(labels)
    plt.colorbar(im1, ax=ax1)
    for i in range(4):
        for j in range(4):
            ax1.text(j, i, f'{real_part[i, j]:.3f}', ha='center', va='center', fontsize=9)
    
    imag_part = np.imag(rho)
    im2 = ax2.imshow(imag_part, cmap='RdBu', vmin=-0.5, vmax=0.5)
    ax2.set_title('Imaginary Part')
    ax2.set_xticks(range(4))
    ax2.set_yticks(range(4))
    ax2.set_xticklabels(labels)
    ax2.set_yticklabels(labels)
    plt.colorbar(im2, ax=ax2)
    for i in range(4):
        for j in range(4):
            ax2.text(j, i, f'{imag_part[i, j]:.3f}', ha='center', va='center', fontsize=9)
    
    plt.tight_layout()
    return fig

# ================================================================
# MAIN
# ================================================================

ideal = bell_state_phi_plus()
true_state = noisy_state(ideal, 0.95)
projectors = create_projectors()
measurements = simulate_measurements(true_state, projectors, 10000)
rho_recon = reconstruct_density_matrix(measurements, projectors)

fid = calculate_fidelity(rho_recon, ideal)
conc = calculate_concurrence(rho_recon)
bell_s = calculate_bell_parameter(rho_recon)

print(f"\nFidelity: {fid:.4f}")
print(f"Concurrence: {conc:.4f}")
print(f"Bell S: {bell_s:.4f}")

print("\nReconstructed Density Matrix:")
print("Real part:")
print(np.real(rho_recon))
print("\nImaginary part:")
print(np.imag(rho_recon))

fig = plot_density_matrix(rho_recon, f"Reconstructed State (F={fid:.3f})")
plt.savefig('quantum_tomography_result.png', dpi=300, bbox_inches='tight')
plt.savefig('quantum_tomography_result.pdf', bbox_inches='tight')
plt.close()

print("\n✓ Saved: quantum_tomography_result.png/pdf")