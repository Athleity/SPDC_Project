"""
STEP 3: Compare PsiPlus vs PsiMinus
"""

import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt

print("="*70)
print("STEP 3: COMPARING PSIPLUS VS PSIMINUS")
print("="*70)

data_path = "D:/SPDC_Project/03_Data/ECMBI_Tomography"
files = [f for f in os.listdir(data_path) if f.endswith('.csv')]

psiplus_data = []
psiminus_data = []

for file in files:
    match = re.search(r'(\d+)mW', file)
    power = int(match.group(1)) if match else 0
    
    df = pd.read_csv(os.path.join(data_path, file))
    
    hh = df[df['basis'] == 'h,h']['TT'].values[0] if len(df[df['basis'] == 'h,h']) > 0 else 0
    hv = df[df['basis'] == 'h,v']['TT'].values[0] if len(df[df['basis'] == 'h,v']) > 0 else 0
    vh = df[df['basis'] == 'v,h']['TT'].values[0] if len(df[df['basis'] == 'v,h']) > 0 else 0
    vv = df[df['basis'] == 'v,v']['TT'].values[0] if len(df[df['basis'] == 'v,v']) > 0 else 0
    
    total = hh + hv + vh + vv
    fidelity = (hh + vv) / total if total > 0 else 0
    visibility = (hh + vv - hv - vh) / total if total > 0 else 0
    
    if "PSIPLUS" in file.upper():
        psiplus_data.append({'power': power, 'fidelity': fidelity, 'visibility': visibility})
    else:
        psiminus_data.append({'power': power, 'fidelity': fidelity, 'visibility': visibility})

df_plus = pd.DataFrame(psiplus_data).sort_values('power')
df_minus = pd.DataFrame(psiminus_data).sort_values('power')

print("\nPsiPlus (|Φ⁺⟩) results:")
print(df_plus.to_string(index=False))
print("\nPsiMinus (|Ψ⁻⟩) results:")
print(df_minus.to_string(index=False))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(df_plus['power'], df_plus['fidelity'], 'bo-', linewidth=2, markersize=8, label='PsiPlus')
ax1.plot(df_minus['power'], df_minus['fidelity'], 'rs-', linewidth=2, markersize=8, label='PsiMinus')
ax1.set_xlabel('Pump Power (mW)', fontsize=12)
ax1.set_ylabel('Fidelity', fontsize=12)
ax1.set_title('(a) Fidelity Comparison', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(df_plus['power'], df_plus['visibility'], 'bo-', linewidth=2, markersize=8, label='PsiPlus')
ax2.plot(df_minus['power'], df_minus['visibility'], 'rs-', linewidth=2, markersize=8, label='PsiMinus')
ax2.set_xlabel('Pump Power (mW)', fontsize=12)
ax2.set_ylabel('Visibility', fontsize=12)
ax2.set_title('(b) Visibility Comparison', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('state_comparison.png', dpi=300)
plt.savefig('state_comparison.pdf')
plt.close()

print("\n✓ Saved: state_comparison.png/pdf")

mean_plus = df_plus['fidelity'].mean()
mean_minus = df_minus['fidelity'].mean()
std_plus = df_plus['fidelity'].std()
std_minus = df_minus['fidelity'].std()

print("\n" + "="*70)
print("STATISTICAL SUMMARY")
print("="*70)
print(f"PsiPlus:  Fidelity = {mean_plus:.4f} ± {std_plus:.4f}")
print(f"PsiMinus: Fidelity = {mean_minus:.4f} ± {std_minus:.4f}")
print(f"Difference: {abs(mean_plus - mean_minus):.4f}")

if abs(mean_plus - mean_minus) < 0.01:
    print("\nConclusion: Both Bell states show similar fidelity.")
else:
    print("\nConclusion: There is a measurable difference between the states.")

print("="*70)