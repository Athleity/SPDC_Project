"""
Power Prediction from Coincidence Counts
Strong ML result for thesis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import os
import re

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 11

print("="*70)
print("POWER PREDICTION FROM SPDC COINCIDENCE DATA")
print("="*70)

# Load data
path = "D:/SPDC_Project/03_Data/ECMBI_Tomography"
files = [f for f in os.listdir(path) if f.endswith('.csv')]

all_tt = []
all_powers = []

for file in files:
    match = re.search(r'(\d+)mW', file)
    power = int(match.group(1)) if match else 0
    df = pd.read_csv(os.path.join(path, file))
    all_tt.extend(df['TT'].values)
    all_powers.extend([power] * len(df))

X = np.array(all_tt).reshape(-1, 1)
y = np.array(all_powers)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nResults:")
print(f"  R² Score: {r2:.4f}")
print(f"  RMSE: {rmse:.2f} mW")

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, y_pred, alpha=0.5, c='blue')
ax.plot([0, 280], [0, 280], 'r--', label='Ideal Prediction')
ax.set_xlabel('Actual Pump Power (mW)')
ax.set_ylabel('Predicted Pump Power (mW)')
ax.set_title(f'SPDC Power Prediction from Coincidence Counts\n(R² = {r2:.3f}, RMSE = {rmse:.1f} mW)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('power_prediction_result.png', dpi=300)
plt.savefig('power_prediction_result.pdf')
plt.close()

print("\n✓ Saved: power_prediction_result.png/pdf")
print("="*70)