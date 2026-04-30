"""
ML POWER PREDICTION - Random Forest to predict pump power from coincidences
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import os
import re

print("="*70)
print("ML POWER PREDICTION FROM SPDC DATA")
print("="*70)

# Load your ECMBI data
data_path = "D:/SPDC_Project/03_Data/ECMBI_Tomography"
files = [f for f in os.listdir(data_path) if f.endswith('.csv')]

print(f"\nFound {len(files)} CSV files")

# Build dataset
X_list = []  # Features (TT values for each basis)
y_list = []  # Target (pump power in mW)

for file in files:
    # Extract pump power from filename
    match = re.search(r'(\d+)mW', file)
    power = int(match.group(1)) if match else 0
    
    df = pd.read_csv(os.path.join(data_path, file))
    tt_values = df['TT'].values
    
    X_list.append(tt_values)
    y_list.append(power)

X = np.array(X_list)
y = np.array(y_list)

print(f"Feature matrix shape: {X.shape}")
print(f"Target values: {np.unique(y)} mW")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Train Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)

# Metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nRandom Forest Results:")
print(f"  R² Score: {r2:.4f}")
print(f"  RMSE: {rmse:.2f} mW")

# Cross-validation
cv_scores = cross_val_score(rf, X, y, cv=3)
print(f"  Cross-validation R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, s=50)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2, label='Perfect prediction')
plt.xlabel('Actual Pump Power (mW)', fontsize=12)
plt.ylabel('Predicted Pump Power (mW)', fontsize=12)
plt.title(f'Random Forest: Power Prediction\nR² = {r2:.3f}, RMSE = {rmse:.1f} mW', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('power_prediction_result.png', dpi=300)
plt.savefig('power_prediction_result.pdf')
plt.close()

print("\n✓ Saved: power_prediction_result.png/pdf")

# Feature importance
feature_imp = pd.DataFrame({
    'feature': range(X.shape[1]),
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 10 most important features (basis indices):")
for i in range(min(10, len(feature_imp))):
    print(f"  Basis {feature_imp.iloc[i]['feature']}: {feature_imp.iloc[i]['importance']:.4f}")

print("\n" + "="*70)
print("✅ POWER PREDICTION COMPLETE")
print("="*70)