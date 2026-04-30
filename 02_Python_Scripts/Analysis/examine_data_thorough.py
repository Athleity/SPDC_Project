"""
THOROUGH DATA EXAMINATION
Check what's really in the ECMBI dataset
"""

import pandas as pd
import numpy as np
import os
import re

print("="*60)
print("THOROUGH DATA EXAMINATION")
print("="*60)

# Path to data
path = "D:/SPDC_Project/03_Data/ECMBI_Tomography"
files = [f for f in os.listdir(path) if f.endswith('.csv')]

print(f"\nFound {len(files)} files")

# Load all files and examine structure
all_bases = []
all_TT = []
all_powers = []
all_states = []

for file in files:
    # Get state and power
    if 'PSIPLUS' in file.upper():
        state = 'PsiPlus'
    else:
        state = 'PsiMinus'
    
    match = re.search(r'(\d+)mW', file)
    power = int(match.group(1)) if match else 0
    
    df = pd.read_csv(os.path.join(path, file))
    
    print(f"\n{file}")
    print(f"  State: {state}, Power: {power} mW")
    print(f"  Rows: {len(df)}, Columns: {list(df.columns)}")
    print(f"  Basis values: {df['basis'].unique()}")
    print(f"  TT range: {df['TT'].min()} to {df['TT'].max()}")
    
    all_TT.extend(df['TT'].values)
    all_bases.extend(df['basis'].values)
    all_powers.extend([power] * len(df))
    all_states.extend([state] * len(df))

print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

# Create DataFrame
summary_df = pd.DataFrame({
    'basis': all_bases,
    'TT': all_TT,
    'power_mW': all_powers,
    'state': all_states
})

print(f"\nTotal samples: {len(summary_df)}")
print(f"Unique bases: {summary_df['basis'].nunique()}")
print(f"Bases: {summary_df['basis'].unique()}")

print(f"\nTT Statistics by Power:")
for p in sorted(summary_df['power_mW'].unique()):
    subset = summary_df[summary_df['power_mW'] == p]
    print(f"  {p} mW: mean TT = {subset['TT'].mean():.0f}, std = {subset['TT'].std():.0f}")

print(f"\nTT Statistics by State:")
for s in summary_df['state'].unique():
    subset = summary_df[summary_df['state'] == s]
    print(f"  {s}: mean TT = {subset['TT'].mean():.0f}, std = {subset['TT'].std():.0f}")

# Check if TT alone can predict state
print("\n" + "="*60)
print("CAN TT ALONE DISTINGUISH STATES?")
print("="*60)

psiplus_tt = summary_df[summary_df['state'] == 'PsiPlus']['TT'].values
psiminus_tt = summary_df[summary_df['state'] == 'PsiMinus']['TT'].values

print(f"PsiPlus TT range: {psiplus_tt.min()} to {psiplus_tt.max()}")
print(f"PsiMinus TT range: {psiminus_tt.min()} to {psiminus_tt.max()}")

# Check overlap
overlap = len(set(psiplus_tt) & set(psiminus_tt))
print(f"Overlap between states: {overlap} values")

if psiplus_tt.mean() == psiminus_tt.mean():
    print("\n❌ TT means are IDENTICAL! This explains why classification failed.")
else:
    print(f"\nPsiPlus mean TT: {psiplus_tt.mean():.0f}")
    print(f"PsiMinus mean TT: {psiminus_tt.mean():.0f}")