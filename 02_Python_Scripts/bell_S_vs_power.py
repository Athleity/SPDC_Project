"""
STEP 2: Bell parameter S vs pump power
"""

import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt

print("="*70)
print("STEP 2: BELL PARAMETER S VS PUMP POWER")
print("="*70)

data_path = "D:/SPDC_Project/03_Data/ECMBI_Tomography"
files = [f for f in os.listdir(data_path) if f.endswith('.csv')]

results = []

for file in files:
    match = re.search(r'(\d+)mW', file)
    power = int(match.group(1)) if match else 0
    
    state = "PsiPlus" if "PSIPLUS" in file.upper() else "PsiMinus"
    
    df = pd.read_csv(os.path.join(data_path, file))
    
    hh = df[df['basis'] == 'h,h']['TT'].values[0] if len(df[df['basis'] == 'h,h']) > 0 else 0
    hv = df[df['basis'] == 'h,v']['TT'].values[0] if len(df[df['basis'] == 'h,v']) > 0 else 0
    vh = df[df['basis'] == 'v,h']['TT'].values[0] if len(df[df['basis'] == 'v,h']) > 0 else 0
    vv = df[df['basis'] == 'v,v']['TT'].values[0] if len(df[df['basis'] == 'v,v']) > 0 else 0
    
    total = hh + hv + vh + vv
    
    if total > 0:
        E_hv = (hh + vv - hv - vh) / total
    else:
        E_hv = 0
    
    S_estimate = 2.828 * abs(E_hv)
    
    results.append({
        'power_mW': power,
        'state': state,
        'S_parameter': S_estimate,
        'correlation_E': E_hv,
        'visibility': (hh + vv - hv - vh) / total if total > 0 else 0
    })

df_results = pd.DataFrame(results)
avg_by_power = df_results.groupby('power_mW').agg({
    'S_parameter': 'mean',
    'correlation_E': 'mean',
    'visibility': 'mean'
}).reset_index()

print("\nBell parameter S vs pump power:")
print(avg_by_power.to_string(index=False))

avg_by_power.to_csv('bell_S_vs_power.csv', index=False)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(avg_by_power['power_mW'], avg_by_power['S_parameter'], 'ro-', linewidth=2.5, markersize=8)
ax.axhline(y=2, color='b', linestyle='--', linewidth=2, label='Classical limit (S=2)')
ax.axhline(y=2.828, color='g', linestyle=':', linewidth=2, label='Quantum limit (2√2 = 2.828)')
ax.set_xlabel('Pump Power (mW)', fontsize=14)
ax.set_ylabel('Bell Parameter S', fontsize=14)
ax.set_title('Bell Parameter S vs Pump Power', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bell_S_vs_power.png', dpi=300, bbox_inches='tight')
plt.savefig('bell_S_vs_power.pdf', bbox_inches='tight')
plt.close()

print("\n✓ Saved: bell_S_vs_power.png/pdf")

max_row = avg_by_power.loc[avg_by_power['S_parameter'].idxmax()]
print(f"\nMaximum Bell parameter S = {max_row['S_parameter']:.3f} at {max_row['power_mW']:.0f} mW")
print("="*70)