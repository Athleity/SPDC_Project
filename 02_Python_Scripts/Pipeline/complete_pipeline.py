

import subprocess
import sys
import os

print("="*70)
print("COMPLETE PIPELINE - RUNNING ALL SIMULATIONS")
print("="*70)

python_path = r"C:\Users\priya\AppData\Local\Programs\Python\Python312\python.exe"
base_dir = r"D:\SPDC_Project\02_Python_Scripts"

scripts = [
    ("Main", "quantum_tomography.py"),
    ("Main", "spdc_master_equation.py"),
    ("Main", "bell_inequality.py"),
    ("Visualization", "entanglement_witness.py"),
    ("Visualization", "quantum_state_visualization.py"),
    ("Analysis", "experimental_data_fitter.py"),
]

for folder, script in scripts:
    script_path = os.path.join(base_dir, folder, script)
    print(f"\n{'='*70}")
    print(f"Running: {script}")
    print('='*70)
    
    if os.path.exists(script_path):
        result = subprocess.run([python_path, script_path], capture_output=False)
        if result.returncode != 0:
            print(f"Warning: {script} had errors")
    else:
        print(f"File not found: {script_path}")

print("\n" + "="*70)
print("✅ ALL SIMULATIONS COMPLETE")
print("="*70)
print("\nGenerated files are in D:\\SPDC_Project\\05_Results\\")
print("="*70)