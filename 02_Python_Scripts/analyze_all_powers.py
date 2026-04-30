"""
================================================================================
STEP 1: ANALYZE TOMOGRAPHY DATA AT ALL PUMP POWERS
================================================================================
This script processes your ECMBI tomography data files and calculates:
1. Fidelity at each pump power (20, 50, 100, 150, 200, 250 mW)
2. Coincidence counts (HH, VV, HV, VH) at each power
3. Creates a plot of fidelity vs pump power for your thesis

WHAT YOU NEED:
- Your ECMBI tomography CSV files in D:\SPDC_Project\03_Data\ECMBI_Tomography
- The script will find them automatically

WHAT YOU GET:
- fidelity_vs_power.csv (data table)
- fidelity_vs_power.png (thesis figure)
- fidelity_vs_power.pdf (vector figure for thesis)

================================================================================
"""

# Import required libraries
import pandas as pd      # For handling CSV files and data tables
import numpy as np       # For numerical calculations
import os                # For finding files in folders
import re                # For extracting numbers from filenames
import matplotlib.pyplot as plt  # For creating graphs

print("="*70)
print("STEP 1: ANALYZING TOMOGRAPHY DATA AT ALL PUMP POWERS")
print("="*70)
print("\nThis script will process your ECMBI tomography files")
print("and calculate fidelity for each pump power.\n")

# ============================================================================
# PART 1: FIND ALL TOMOGRAPHY FILES
# ============================================================================

# Tell the script where to find your data
data_path = "D:/SPDC_Project/03_Data/ECMBI_Tomography"

# Get a list of all CSV files in that folder
all_files = os.listdir(data_path)

# Keep only files that are CSV (ends with .csv)
csv_files = [f for f in all_files if f.endswith('.csv')]

print(f"Found {len(csv_files)} CSV files in the folder.")

# ============================================================================
# PART 2: CREATE A LIST TO STORE RESULTS
# ============================================================================

# This will hold all our results
results = []

# ============================================================================
# PART 3: PROCESS EACH FILE ONE BY ONE
# ============================================================================

# Loop through each file
for file in csv_files:
    
    # --- Step 3.1: Extract pump power from filename ---
    # Filename example: "PSIPLUStwoQBtomo_100mW_2023-02-10--11h-26m_10--11h-34mS1.csv"
    # We need to find "100mW" and extract the number 100
    
    # Search for pattern: number followed by "mW"
    match = re.search(r'(\d+)mW', file)
    
    if match:
        power = int(match.group(1))  # Convert to integer (e.g., 100)
    else:
        power = 0  # If no power found, set to 0
    
    # --- Step 3.2: Determine which Bell state it is ---
    if "PSIPLUS" in file.upper():
        state = "PsiPlus (|Φ⁺⟩)"
    elif "PSIMINUS" in file.upper():
        state = "PsiMinus (|Ψ⁻⟩)"
    else:
        state = "Unknown"
    
    # --- Step 3.3: Load the CSV file ---
    file_path = os.path.join(data_path, file)
    df = pd.read_csv(file_path)
    
    # --- Step 3.4: Extract coincidence counts for different measurement bases ---
    # 'basis' column contains values like 'h,h', 'h,v', 'v,h', 'v,v'
    # 'TT' column contains two-fold coincidence counts
    
    # Find HH coincidences (both photons horizontal)
    hh_rows = df[df['basis'] == 'h,h']
    if len(hh_rows) > 0:
        HH = hh_rows['TT'].values[0]
    else:
        HH = 0
    
    # Find HV coincidences (first horizontal, second vertical)
    hv_rows = df[df['basis'] == 'h,v']
    if len(hv_rows) > 0:
        HV = hv_rows['TT'].values[0]
    else:
        HV = 0
    
    # Find VH coincidences (first vertical, second horizontal)
    vh_rows = df[df['basis'] == 'v,h']
    if len(vh_rows) > 0:
        VH = vh_rows['TT'].values[0]
    else:
        VH = 0
    
    # Find VV coincidences (both vertical)
    vv_rows = df[df['basis'] == 'v,v']
    if len(vv_rows) > 0:
        VV = vv_rows['TT'].values[0]
    else:
        VV = 0
    
    # --- Step 3.5: Calculate total coincidences ---
    total_coincidences = HH + HV + VH + VV
    
    # --- Step 3.6: Calculate fidelity ---
    # For a Bell state, fidelity = (HH + VV) / Total
    # This is because ideal Bell state has only HH and VV, no HV or VH
    if total_coincidences > 0:
        fidelity = (HH + VV) / total_coincidences
    else:
        fidelity = 0
    
    # --- Step 3.7: Store the results ---
    results.append({
        'filename': file,
        'power_mW': power,
        'state': state,
        'HH': HH,
        'HV': HV,
        'VH': VH,
        'VV': VV,
        'total': total_coincidences,
        'fidelity': fidelity
    })
    
    # Print progress
    print(f"  Processed: {power} mW - {state} -> Fidelity = {fidelity:.4f}")

# ============================================================================
# PART 4: CONVERT RESULTS TO A TABLE
# ============================================================================

# Create a pandas DataFrame (like an Excel table)
df_results = pd.DataFrame(results)

# Sort by pump power (smallest to largest)
df_results = df_results.sort_values('power_mW')

print("\n" + "="*70)
print("RESULTS TABLE")
print("="*70)
print(df_results[['power_mW', 'state', 'fidelity', 'HH', 'VV', 'HV', 'VH']].to_string(index=False))

# ============================================================================
# PART 5: CALCULATE AVERAGE FOR EACH PUMP POWER (over both states)
# ============================================================================

# Group by pump power and calculate average fidelity
avg_by_power = df_results.groupby('power_mW').agg({
    'fidelity': 'mean',
    'HH': 'mean',
    'VV': 'mean',
    'HV': 'mean',
    'VH': 'mean'
}).reset_index()

print("\n" + "="*70)
print("AVERAGE FIDELITY BY PUMP POWER")
print("="*70)
print(avg_by_power[['power_mW', 'fidelity']].to_string(index=False))

# ============================================================================
# PART 6: SAVE RESULTS TO CSV FILE
# ============================================================================

# Save individual results
df_results.to_csv('tomography_all_powers.csv', index=False)
print("\n✓ Saved: tomography_all_powers.csv")

# Save averaged results
avg_by_power.to_csv('fidelity_vs_power.csv', index=False)
print("✓ Saved: fidelity_vs_power.csv")

# ============================================================================
# PART 7: CREATE A PUBLICATION-QUALITY PLOT
# ============================================================================

print("\n" + "="*70)
print("GENERATING PLOT")
print("="*70)

# Create a figure (window for the plot)
plt.figure(figsize=(10, 6))

# Plot average fidelity vs pump power
plt.plot(avg_by_power['power_mW'], avg_by_power['fidelity'], 
         'bo-',           # b=blue, o=circle markers, -=solid line
         linewidth=2.5,   # Thick line
         markersize=8,    # Size of circles
         label='Average Fidelity')

# Also plot individual states (optional, shows both)
# Separate PsiPlus and PsiMinus
psiplus_data = df_results[df_results['state'] == "PsiPlus (|Φ⁺⟩)"]
psiminus_data = df_results[df_results['state'] == "PsiMinus (|Ψ⁻⟩)"]

plt.plot(psiplus_data['power_mW'], psiplus_data['fidelity'], 
         'gs--',          # g=green, s=square, --=dashed line
         linewidth=1.5,
         markersize=6,
         label='PsiPlus (|Φ⁺⟩)')

plt.plot(psiminus_data['power_mW'], psiminus_data['fidelity'], 
         'r^--',          # r=red, ^=triangle, --=dashed line
         linewidth=1.5,
         markersize=6,
         label='PsiMinus (|Ψ⁻⟩)')

# Add a horizontal line at fidelity = 0.89 (threshold from your paper)
plt.axhline(y=0.89, color='purple', linestyle=':', linewidth=2, 
            label='Target Fidelity (0.89 from paper)')

# Labels and title
plt.xlabel('Pump Power (mW)', fontsize=14, fontweight='bold')
plt.ylabel('Fidelity', fontsize=14, fontweight='bold')
plt.title('Fidelity vs Pump Power - ECMBI Experimental Data', fontsize=16, fontweight='bold')

# Add legend
plt.legend(loc='best', fontsize=11)

# Add grid for easier reading
plt.grid(True, alpha=0.3, linestyle='--')

# Set y-axis limits (fidelity between 0.85 and 1.0)
plt.ylim(0.85, 1.0)

# Save the plot as PNG (for PowerPoint)
plt.savefig('fidelity_vs_power.png', dpi=300, bbox_inches='tight')
print("✓ Saved: fidelity_vs_power.png")

# Save as PDF (for thesis - vector quality)
plt.savefig('fidelity_vs_power.pdf', bbox_inches='tight')
print("✓ Saved: fidelity_vs_power.pdf")

# Also show the plot on screen (optional)
# plt.show()

# ============================================================================
# PART 8: PRINT SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

# Find the best fidelity (highest value)
best_row = avg_by_power.loc[avg_by_power['fidelity'].idxmax()]
print(f"\nBest fidelity: {best_row['fidelity']:.4f} at {best_row['power_mW']:.0f} mW")

# Find the worst fidelity (lowest value)
worst_row = avg_by_power.loc[avg_by_power['fidelity'].idxmin()]
print(f"Worst fidelity: {worst_row['fidelity']:.4f} at {worst_row['power_mW']:.0f} mW")

# Calculate average fidelity across all powers
avg_fidelity_all = avg_by_power['fidelity'].mean()
print(f"Average fidelity across all powers: {avg_fidelity_all:.4f}")

# ============================================================================
# PART 9: WHAT THESE RESULTS MEAN
# ============================================================================

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)
print("""
Fidelity measures how close your quantum state is to the ideal Bell state.
- Fidelity = 1.00 means perfect Bell state
- Fidelity = 0.89 is the threshold for Bell inequality violation (from your paper)
- Higher fidelity = better entanglement

Your results show:
""")

for _, row in avg_by_power.iterrows():
    power = row['power_mW']
    fid = row['fidelity']
    if fid > 0.89:
        status = "✓ Good (violates Bell inequality)"
    else:
        status = "✗ Below threshold"
    print(f"  {power:3d} mW: Fidelity = {fid:.4f} - {status}")

print("\n" + "="*70)
print("STEP 1 COMPLETE!")
print("="*70)
print("\nFiles created:")
print("  1. tomography_all_powers.csv - Raw data for all files")
print("  2. fidelity_vs_power.csv - Averaged data by pump power")
print("  3. fidelity_vs_power.png - Figure for your thesis")
print("  4. fidelity_vs_power.pdf - Vector figure for your thesis")
print("\nThese figures show how entanglement quality changes with pump power.")
print("="*70)