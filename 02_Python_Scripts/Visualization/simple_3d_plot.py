import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print("="*70)
print("SIMPLE 3D DENSITY MATRIX PLOT")
print("="*70)

# ================================================================
# CORRECT DENSITY MATRIX VALUES (From your simulation)
# ================================================================

# Real part of the density matrix
real_part = np.array([
    [0.4906, 0.0000, 0.0000, 0.4811],
    [0.0000, 0.0095, 0.0000, 0.0000],
    [0.0000, 0.0000, 0.0095, 0.0000],
    [0.4811, 0.0000, 0.0000, 0.4906]
])

# Imaginary part (all zeros for Bell state)
imag_part = np.zeros((4, 4))

labels = ['HH', 'HV', 'VH', 'VV']

print("\nReal part of density matrix:")
print(real_part)
print("\nImaginary part:")
print(imag_part)

# ================================================================
# CREATE HEATMAP
# ================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Real part heatmap
im1 = ax1.imshow(real_part, cmap='RdBu', vmin=-0.1, vmax=0.6)
ax1.set_xticks(range(4))
ax1.set_yticks(range(4))
ax1.set_xticklabels(labels)
ax1.set_yticklabels(labels)
ax1.set_title('Real Part of Density Matrix')
plt.colorbar(im1, ax=ax1)

# Add numbers
for i in range(4):
    for j in range(4):
        ax1.text(j, i, f'{real_part[i, j]:.4f}', 
                ha='center', va='center', fontsize=9)

# Imaginary part heatmap
im2 = ax2.imshow(imag_part, cmap='RdBu', vmin=-0.1, vmax=0.1)
ax2.set_xticks(range(4))
ax2.set_yticks(range(4))
ax2.set_xticklabels(labels)
ax2.set_yticklabels(labels)
ax2.set_title('Imaginary Part of Density Matrix')
plt.colorbar(im2, ax=ax2)

for i in range(4):
    for j in range(4):
        ax2.text(j, i, f'{imag_part[i, j]:.4f}', 
                ha='center', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('density_matrix_heatmap.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: density_matrix_heatmap.png")

# ================================================================
# CREATE 3D BAR PLOT
# ================================================================

fig = plt.figure(figsize=(14, 6))

# Real part 3D
ax1 = fig.add_subplot(121, projection='3d')
xpos, ypos = np.meshgrid(range(4), range(4))
xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = np.zeros_like(xpos)
dx = dy = 0.7
dz = real_part.flatten()

# Color bars based on value
colors = plt.cm.viridis((dz - np.min(dz)) / (np.max(dz) - np.min(dz) + 0.001))
ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, alpha=0.9)
ax1.set_xticks(range(4))
ax1.set_yticks(range(4))
ax1.set_xticklabels(labels)
ax1.set_yticklabels(labels)
ax1.set_zlabel('Value')
ax1.set_title('Real Part of Density Matrix (3D View)')
ax1.view_init(elev=25, azim=-60)

# Imaginary part 3D
ax2 = fig.add_subplot(122, projection='3d')
dz_imag = imag_part.flatten()
ax2.bar3d(xpos, ypos, zpos, dx, dy, dz_imag, color='lightblue', alpha=0.9)
ax2.set_xticks(range(4))
ax2.set_yticks(range(4))
ax2.set_xticklabels(labels)
ax2.set_yticklabels(labels)
ax2.set_zlabel('Value')
ax2.set_title('Imaginary Part of Density Matrix (3D View)')
ax2.view_init(elev=25, azim=-60)

plt.tight_layout()
plt.savefig('density_matrix_3d.png', dpi=300, bbox_inches='tight')
print("✓ Saved: density_matrix_3d.png")

# ================================================================
# CREATE BAR CHART OF DIAGONAL ELEMENTS
# ================================================================

fig, ax = plt.subplots(figsize=(8, 5))
diagonal = [real_part[0,0], real_part[1,1], real_part[2,2], real_part[3,3]]
bars = ax.bar(labels, diagonal, color=['blue', 'orange', 'orange', 'blue'], alpha=0.7)
ax.set_ylabel('Probability')
ax.set_title('Diagonal Elements of Density Matrix')
ax.set_ylim(0, 0.6)

# Add value labels
for bar, val in zip(bars, diagonal):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
            f'{val:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('density_matrix_diagonal.png', dpi=300, bbox_inches='tight')
print("✓ Saved: density_matrix_diagonal.png")

# ================================================================
# PRINT SUMMARY
# ================================================================

print("\n" + "="*70)
print("SUMMARY - DENSITY MATRIX ELEMENTS")
print("="*70)
print(f"\nDiagonal elements (probabilities):")
print(f"  |HH⟩: {real_part[0,0]:.4f} ({real_part[0,0]*100:.1f}%)")
print(f"  |HV⟩: {real_part[1,1]:.4f} ({real_part[1,1]*100:.1f}%)")
print(f"  |VH⟩: {real_part[2,2]:.4f} ({real_part[2,2]*100:.1f}%)")
print(f"  |VV⟩: {real_part[3,3]:.4f} ({real_part[3,3]*100:.1f}%)")

print(f"\nCoherence terms (off-diagonal):")
print(f"  ⟨HH|ρ|VV⟩: {real_part[0,3]:.4f}")
print(f"  ⟨VV|ρ|HH⟩: {real_part[3,0]:.4f}")

print("\n" + "="*70)
print("✅ All plots saved!")
print("Files created:")
print("  - density_matrix_heatmap.png")
print("  - density_matrix_3d.png")
print("  - density_matrix_diagonal.png")
print("="*70)