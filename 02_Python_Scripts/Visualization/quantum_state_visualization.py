import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print("="*70)
print("PhD-LEVEL: QUANTUM STATE VISUALIZATION - FIXED")
print("="*70)

# ================================================================
# CREATE BELL STATE
# ================================================================

def bell_state_phi_plus():
    """|Φ⁺⟩ = (|HH⟩ + |VV⟩)/√2"""
    rho = np.zeros((4, 4), dtype=complex)
    rho[0, 0] = 0.5
    rho[3, 3] = 0.5
    rho[0, 3] = 0.5
    rho[3, 0] = 0.5
    return rho

def noisy_state(rho_ideal, fidelity):
    """ρ = fidelity * ρ_ideal + (1-fidelity) * I/4"""
    mixed = np.eye(4, dtype=complex) / 4
    return fidelity * rho_ideal + (1 - fidelity) * mixed

# Create state
fidelity = 0.9622
rho = noisy_state(bell_state_phi_plus(), fidelity)

print(f"\nState fidelity: {fidelity:.4f}")

# ================================================================
# DENSITY MATRIX VALUES
# ================================================================

real_part = np.real(rho)
imag_part = np.imag(rho)

print("\nReal part:")
print(real_part)
print("\nImaginary part:")
print(imag_part)

# ================================================================
# HEATMAP (with values)
# ================================================================

print("\nGenerating heatmap...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

labels = ['HH', 'HV', 'VH', 'VV']

# Real part heatmap
im1 = ax1.imshow(real_part, cmap='RdBu', vmin=-0.1, vmax=0.6)
ax1.set_xticks(range(4))
ax1.set_yticks(range(4))
ax1.set_xticklabels(labels)
ax1.set_yticklabels(labels)
ax1.set_title(f'Real Part (Fidelity = {fidelity:.3f})')
plt.colorbar(im1, ax=ax1)

# Add numbers in heatmap
for i in range(4):
    for j in range(4):
        ax1.text(j, i, f'{real_part[i, j]:.4f}', 
                ha='center', va='center', fontsize=9, color='black')

# Imaginary part heatmap
im2 = ax2.imshow(imag_part, cmap='RdBu', vmin=-0.1, vmax=0.1)
ax2.set_xticks(range(4))
ax2.set_yticks(range(4))
ax2.set_xticklabels(labels)
ax2.set_yticklabels(labels)
ax2.set_title('Imaginary Part')
plt.colorbar(im2, ax=ax2)

for i in range(4):
    for j in range(4):
        ax2.text(j, i, f'{imag_part[i, j]:.4f}', 
                ha='center', va='center', fontsize=9, color='black')

plt.tight_layout()
plt.savefig('quantum_state_heatmap.png', dpi=300, bbox_inches='tight')
plt.savefig('quantum_state_heatmap.pdf', bbox_inches='tight')
plt.close()
print("✓ Saved: quantum_state_heatmap.png/pdf")

# ================================================================
# 3D BAR PLOT (NO TEXT OVERLAY)
# ================================================================

print("\nGenerating 3D plot (no text overlay)...")

fig = plt.figure(figsize=(14, 6))

# Real part 3D
ax1 = fig.add_subplot(121, projection='3d')
xpos, ypos = np.meshgrid(range(4), range(4))
xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = np.zeros_like(xpos)
dx = dy = 0.7
dz = real_part.flatten()

colors = plt.cm.viridis((dz - np.min(dz)) / (np.max(dz) - np.min(dz) + 0.001))
ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, alpha=0.9)
ax1.set_xticks(range(4))
ax1.set_yticks(range(4))
ax1.set_xticklabels(['HH', 'HV', 'VH', 'VV'])
ax1.set_yticklabels(['HH', 'HV', 'VH', 'VV'])
ax1.set_zlabel('Value')
ax1.set_title('Real Part of Density Matrix')
ax1.view_init(elev=25, azim=-60)

# Imaginary part 3D
ax2 = fig.add_subplot(122, projection='3d')
dz_imag = imag_part.flatten()
ax2.bar3d(xpos, ypos, zpos, dx, dy, dz_imag, color='lightblue', alpha=0.9)
ax2.set_xticks(range(4))
ax2.set_yticks(range(4))
ax2.set_xticklabels(['HH', 'HV', 'VH', 'VV'])
ax2.set_yticklabels(['HH', 'HV', 'VH', 'VV'])
ax2.set_zlabel('Value')
ax2.set_title('Imaginary Part of Density Matrix')
ax2.view_init(elev=25, azim=-60)

plt.tight_layout()
plt.savefig('quantum_state_3d.png', dpi=300, bbox_inches='tight')
plt.savefig('quantum_state_3d.pdf', bbox_inches='tight')
plt.close()
print("✓ Saved: quantum_state_3d.png/pdf (no text overlap)")

# ================================================================
# DIAGONAL BAR CHART
# ================================================================

print("\nGenerating diagonal bar chart...")

fig, ax = plt.subplots(figsize=(8, 5))
diagonal = [real_part[0,0], real_part[1,1], real_part[2,2], real_part[3,3]]
bars = ax.bar(labels, diagonal, color=['blue', 'orange', 'orange', 'blue'], alpha=0.7)
ax.set_ylabel('Probability')
ax.set_title('Diagonal Elements of Density Matrix')
ax.set_ylim(0, 0.6)

for bar, val in zip(bars, diagonal):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
            f'{val:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('density_matrix_diagonal.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: density_matrix_diagonal.png")

# ================================================================
# SUMMARY
# ================================================================

print("\n" + "="*70)
print("✅ QUANTUM STATE VISUALIZATION COMPLETE")
print("="*70)
print(f"""
Density Matrix Diagonal Elements:
  ρ(HH,HH) = {real_part[0,0]:.4f}
  ρ(HV,HV) = {real_part[1,1]:.4f}
  ρ(VH,VH) = {real_part[2,2]:.4f}
  ρ(VV,VV) = {real_part[3,3]:.4f}

Coherence Terms:
  ρ(HH,VV) = {real_part[0,3]:.4f} + {imag_part[0,3]:.4f}i

Files saved:
  - quantum_state_heatmap.png/pdf
  - quantum_state_3d.png/pdf (fixed - no text overlap)
  - density_matrix_diagonal.png
""")
print("="*70)