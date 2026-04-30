"""
MACHINE LEARNING ON REAL ENTANGLEMENT TOMOGRAPHY DATA
Using ECMBI dataset (published experimental data)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
import os
import re
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ML ANALYSIS ON REAL ENTANGLEMENT TOMOGRAPHY DATA")
print("ECMBI Dataset - Published Experimental Results")
print("="*70)

# Path to data
tomography_path = "D:/SPDC_Project/03_Data/ECMBI_Tomography"
files = [f for f in os.listdir(tomography_path) if f.endswith('.csv')]

print(f"\n[1] Loading {len(files)} tomography files...")

# Store data from all files
all_data = []
all_powers = []
all_states = []

for file in files:
    # Determine state (PsiPlus or PsiMinus)
    if 'PSIPLUS' in file.upper():
        state = 'PsiPlus'
    elif 'PSIMINUS' in file.upper():
        state = 'PsiMinus'
    else:
        state = 'Unknown'
    
    # Extract pump power
    match = re.search(r'(\d+)mW', file)
    if match:
        power = int(match.group(1))
    else:
        power = 0
    
    # Load data
    df = pd.read_csv(os.path.join(tomography_path, file))
    
    # Add metadata columns
    df['power_mW'] = power
    df['state'] = state
    df['filename'] = file
    
    all_data.append(df)
    all_powers.append(power)
    all_states.append(state)

# Combine all data
combined_df = pd.concat(all_data, ignore_index=True)

print(f"\n[2] Combined dataset shape: {combined_df.shape}")
print(f"Columns: {list(combined_df.columns)}")

print(f"\n[3] Data summary:")
print(f"  Power range: {min(all_powers)} to {max(all_powers)} mW")
print(f"  States: {set(all_states)}")

# Create one-hot encoding for basis
basis_dummies = pd.get_dummies(combined_df['basis'], prefix='basis')
combined_df = pd.concat([combined_df, basis_dummies], axis=1)

# Features: use TT (two-fold coincidences) and basis dummies
# TT is the two-fold coincidence count - perfect for Bell inequality
feature_cols = ['TT'] + [col for col in basis_dummies.columns]
X = combined_df[feature_cols].fillna(0).values

# Target 1: State classification (PsiPlus vs PsiMinus)
y_state = (combined_df['state'] == 'PsiPlus').astype(int)

# Target 2: Power regression (predict pump power)
y_power = combined_df['power_mW'].values

print(f"\n[4] Feature matrix shape: {X.shape}")
print(f"  State labels: {y_state.sum()} PsiPlus, {len(y_state)-y_state.sum()} PsiMinus")
print(f"  Power range: {y_power.min()} to {y_power.max()} mW")

# ================================================================
# ML MODEL 1: CLASSIFY QUANTUM STATE (PsiPlus vs PsiMinus)
# ================================================================

print("\n" + "="*70)
print("MODEL 1: Quantum State Classification")
print("Predict whether the state is PsiPlus or PsiMinus from TT coincidences")
print("="*70)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_state, test_size=0.3, random_state=42, stratify=y_state
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_scaled, y_train)
rf_pred = rf_clf.predict(X_test_scaled)

print(f"\nRandom Forest Results:")
print(f"  Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
print(f"  Cross-validation: {cross_val_score(rf_clf, X_train_scaled, y_train, cv=5).mean():.4f}")

print(f"\nClassification Report:")
print(classification_report(y_test, rf_pred, target_names=['PsiMinus', 'PsiPlus']))

# ================================================================
# ML MODEL 2: PUMP POWER REGRESSION
# ================================================================

print("\n" + "="*70)
print("MODEL 2: Pump Power Prediction")
print("Predict the pump power from TT coincidences")
print("="*70)

# Train-test split for regression
X_train, X_test, y_train, y_test = train_test_split(
    X, y_power, test_size=0.3, random_state=42
)

# Scale features
scaler_reg = StandardScaler()
X_train_scaled = scaler_reg.fit_transform(X_train)
X_test_scaled = scaler_reg.transform(X_test)

# Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train_scaled, y_train)
y_pred = rf_reg.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nRandom Forest Regressor Results:")
print(f"  RMSE: {rmse:.2f} mW")
print(f"  R2 Score: {r2:.4f}")

# ================================================================
# VISUALIZATION
# ================================================================

print("\n[5] Generating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Confusion matrix
cm = confusion_matrix(y_test, rf_pred)
axes[0, 0].imshow(cm, cmap='Blues')
axes[0, 0].set_xticks([0, 1])
axes[0, 0].set_yticks([0, 1])
axes[0, 0].set_xticklabels(['PsiMinus', 'PsiPlus'])
axes[0, 0].set_yticklabels(['PsiMinus', 'PsiPlus'])
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_ylabel('True')
axes[0, 0].set_title('Confusion Matrix - State Classification')
for i in range(2):
    for j in range(2):
        axes[0, 0].text(j, i, str(cm[i, j]), ha='center', va='center')

# Plot 2: Feature importance
feature_imp = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_clf.feature_importances_
}).sort_values('importance', ascending=False)

axes[0, 1].barh(feature_imp['feature'][:10], feature_imp['importance'][:10], color='green')
axes[0, 1].set_xlabel('Importance')
axes[0, 1].set_title('Top 10 Features for State Classification')

# Plot 3: Actual vs Predicted Power
axes[1, 0].scatter(y_test, y_pred, alpha=0.5)
axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[1, 0].set_xlabel('Actual Power (mW)')
axes[1, 0].set_ylabel('Predicted Power (mW)')
axes[1, 0].set_title(f'Power Prediction (R² = {r2:.3f})')

# Plot 4: Power distribution by state
powers_psiplus = combined_df[combined_df['state'] == 'PsiPlus']['power_mW'].values
powers_psiminus = combined_df[combined_df['state'] == 'PsiMinus']['power_mW'].values
axes[1, 1].hist(powers_psiplus, bins=10, alpha=0.5, label='PsiPlus', color='blue')
axes[1, 1].hist(powers_psiminus, bins=10, alpha=0.5, label='PsiMinus', color='red')
axes[1, 1].set_xlabel('Pump Power (mW)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Power Distribution by Quantum State')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('ecmbi_ml_results.png', dpi=300, bbox_inches='tight')
plt.savefig('ecmbi_ml_results.pdf', bbox_inches='tight')
plt.close()

print("✓ Saved: ecmbi_ml_results.png/pdf")

# ================================================================
# CALCULATE BELL PARAMETER FROM DATA
# ================================================================

print("\n[6] Calculating Bell parameter from experimental data...")

# Group by basis and calculate coincidence rates
basis_groups = combined_df.groupby(['basis', 'state', 'power_mW'])['TT'].mean().reset_index()

print(f"  Found {len(basis_groups)} basis-state-power combinations")

# For Bell parameter, we need E(a,b) = (N++ + N-- - N+- - N-+)/Total
# For PsiPlus: |Φ⁺⟩ = (|HH⟩ + |VV⟩)/√2

# Get HH and VV coincidences
hh_data = combined_df[combined_df['basis'] == 'h,h']['TT'].values
vv_data = combined_df[combined_df['basis'] == 'v,v']['TT'].values

if len(hh_data) > 0 and len(vv_data) > 0:
    correlation = (hh_data.mean() + vv_data.mean()) / (hh_data.mean() + vv_data.mean())
    print(f"  HH-VV correlation: {correlation:.4f}")

print("\n" + "="*70)
print("ML ANALYSIS COMPLETE")
print("="*70)
print(f"""
SUMMARY:
--------
Dataset: {len(files)} tomography files from published experiment
Total samples: {len(combined_df)}
Pump powers: {sorted(set(all_powers))} mW
Quantum states: PsiPlus, PsiMinus

ML Models:
1. State Classification (Random Forest)
   - Accuracy: {accuracy_score(y_test, rf_pred):.4f}
   
2. Power Regression (Random Forest)
   - R² Score: {r2:.4f}
   - RMSE: {rmse:.2f} mW

Files created:
  - ecmbi_ml_results.png/pdf
""")
print("="*70)