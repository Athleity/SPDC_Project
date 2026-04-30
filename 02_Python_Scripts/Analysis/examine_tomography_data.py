"""
EXAMINE ECMBI TOMOGRAPHY DATA
Understand the structure of real experimental entanglement data
"""

import pandas as pd
import numpy as np
import os

print("="*60)
print("ECMBI TOMOGRAPHY DATA EXAMINATION")
print("="*60)

# Path to tomography data
tomography_path = "D:/SPDC_Project/03_Data/ECMBI_Tomography"

# List all files
files = [f for f in os.listdir(tomography_path) if f.endswith('.csv')]
print(f"\nFound {len(files)} tomography files")

# Separate PsiPlus and PsiMinus
psiplus_files = [f for f in files if 'PSIPLUS' in f.upper()]
psiminus_files = [f for f in files if 'PSIMINUS' in f.upper()]

print(f"\nPsiPlus files: {len(psiplus_files)}")
print(f"PsiMinus files: {len(psiminus_files)}")

# Load first file to understand structure
sample_file = os.path.join(tomography_path, files[0])
df = pd.read_csv(sample_file)

print(f"\n{'='*60}")
print(f"Sample file: {files[0]}")
print(f"{'='*60}")
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nColumn names:")
for i, col in enumerate(df.columns):
    print(f"  {i+1}. {col}")

print(f"\nFirst 10 rows:")
print(df.head(10))

print(f"\nData types:")
print(df.dtypes)

# Extract power from filename
import re
for f in files:
    match = re.search(r'(\d+)mW', f)
    if match:
        power = match.group(1)
        print(f"  {f} -> {power} mW")