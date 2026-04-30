import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
import warnings
import os
import sys
from pathlib import Path

warnings.filterwarnings('ignore')

# Allow importing shared `config.py` from `02_Python_Scripts/` when running inside `Analysis/`.
_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

import config  # noqa: E402

config.ensure_dirs(config.RESULTS_GRAPHS_DIR, config.RESULTS_CSV_DIR, config.RESULTS_PDF_DIR)

print("="*70)
print("REAL MACHINE LEARNING: SPDC PHOTON CLASSIFICATION")
print("="*70)

print("\n[1] Loading photon data...")

possible_paths = [
    str(config.PHOTON_TRACE_CSV),
    str(_SCRIPTS_ROOT / "priyansh50us.csv"),
    "priyansh50us.csv",
    str(_SCRIPTS_ROOT.parent / "03_Data" / "priyansh50us.csv"),
    "../03_Data/priyansh50us.csv",
]

file_path = None
for path in possible_paths:
    if os.path.exists(path):
        file_path = path
        break

if file_path is None:
    print("ERROR: Could not find priyansh50us.csv")
    print(f"Please place the file next to the scripts or at: {config.PHOTON_TRACE_CSV}")
    exit()

df = pd.read_csv(file_path, header=None)
df.columns = ['time_s', 'counts']
df = df.reset_index(drop=True)

print("Loaded " + str(len(df)) + " time bins")
print("Total photons: " + str(df['counts'].sum()))
print("Time range: " + str(df['time_s'].min()) + " to " + str(df['time_s'].max()) + " seconds")

print("\n[2] Creating features from photon counting statistics...")

df['log_counts'] = np.log1p(df['counts'])

window = 5
df['mean_5'] = df['counts'].rolling(window=window).mean().fillna(df['counts'])
df['std_5'] = df['counts'].rolling(window=window).std().fillna(0)
df['max_5'] = df['counts'].rolling(window=window).max().fillna(df['counts'])
df['min_5'] = df['counts'].rolling(window=window).min().fillna(df['counts'])

window = 10
df['mean_10'] = df['counts'].rolling(window=window).mean().fillna(df['counts'])
df['std_10'] = df['counts'].rolling(window=window).std().fillna(0)
df['max_10'] = df['counts'].rolling(window=window).max().fillna(df['counts'])

df['ratio_mean_std'] = df['mean_5'] / (df['std_5'] + 0.01)
df['burst_potential'] = df['max_5'] * df['mean_5'] / (df['std_5'] + 0.01)

feature_cols = ['counts', 'log_counts', 'mean_5', 'std_5', 'max_5', 'min_5', 
                'mean_10', 'std_10', 'max_10', 'ratio_mean_std', 'burst_potential']

print("Created " + str(len(feature_cols)) + " features")

print("\n[3] Creating ground truth labels...")

burst_threshold = 20
df['label'] = (df['counts'] > burst_threshold).astype(int)

signal_count = df['label'].sum()
noise_count = len(df) - signal_count
print("Signal bins (SPDC bursts): " + str(signal_count) + " (" + str(round(signal_count/len(df)*100, 1)) + "%)")
print("Noise bins (background): " + str(noise_count) + " (" + str(round(noise_count/len(df)*100, 1)) + "%)")

print("\n[4] Preparing data for ML...")

X = df[feature_cols].fillna(0).values
y = df['label'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

idx = np.arange(len(df))
idx_train, idx_test = train_test_split(idx, test_size=0.3, random_state=42, stratify=y)

X_train = X_scaled[idx_train]
X_test = X_scaled[idx_test]
y_train = y[idx_train]
y_test = y[idx_test]

print("Training samples: " + str(len(X_train)))
print("Test samples: " + str(len(X_test)))
print("Training signal ratio: " + str(round(y_train.mean()*100, 1)) + "%")
print("Test signal ratio: " + str(round(y_test.mean()*100, 1)) + "%")

print("\n[5] Training Support Vector Machine...")

svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
y_prob = svm.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
print("SVM Accuracy: " + str(round(accuracy, 4)))

cv_scores = cross_val_score(svm, X_train, y_train, cv=5)
print("Cross-validation scores: " + str([round(x, 4) for x in cv_scores]))
print("Mean CV accuracy: " + str(round(cv_scores.mean(), 4)) + " +/- " + str(round(cv_scores.std(), 4)))

print("\n[6] Model Performance Metrics...")

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print("ROC AUC: " + str(round(roc_auc, 4)))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Background', 'SPDC Burst']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print("            Predicted")
print("            Noise  Signal")
print("Actual Noise  " + str(cm[0,0]) + "     " + str(cm[0,1]))
print("       Signal " + str(cm[1,0]) + "     " + str(cm[1,1]))

precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
print("Precision: " + str(round(precision, 4)))
print("Recall: " + str(round(recall, 4)))

print("\n[7] Identifying the most important features...")

from sklearn.inspection import permutation_importance
result = permutation_importance(svm, X_test, y_test, n_repeats=10, random_state=42)
feature_importance = pd.DataFrame({'feature': feature_cols, 'importance': result.importances_mean})
feature_importance = feature_importance.sort_values('importance', ascending=False)

print("Top 5 most important features:")
for i in range(min(5, len(feature_importance))):
    print("  " + str(i+1) + ". " + feature_importance.iloc[i]['feature'] + ": " + str(round(feature_importance.iloc[i]['importance'], 4)))

print("\n[8] Generating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

ax = axes[0, 0]
sample = df.iloc[::500]
ax.plot(sample['time_s'], sample['counts'], 'b.', markersize=1)
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Photon Count')
ax.set_title('Original SPDC Data')
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
test_df = df.loc[idx_test].copy()
test_df['predicted'] = y_pred
signal_pred = test_df[test_df['predicted'] == 1]
noise_pred = test_df[test_df['predicted'] == 0]
ax.scatter(noise_pred['time_s'], noise_pred['counts'], c='red', s=1, alpha=0.5, label='Predicted Noise')
ax.scatter(signal_pred['time_s'], signal_pred['counts'], c='green', s=2, alpha=0.7, label='Predicted Signal')
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Photon Count')
ax.set_title('SVM Classification Results (Test Set)')
ax.legend(markerscale=3)
ax.grid(True, alpha=0.3)

ax = axes[0, 2]
ax.plot(fpr, tpr, 'b-', linewidth=2, label='SVM (AUC = ' + str(round(roc_auc, 3)) + ')')
ax.plot([0, 1], [0, 1], 'r--', label='Random Classifier')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.imshow(cm, cmap='Blues')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Noise', 'Signal'])
ax.set_yticklabels(['Noise', 'Signal'])
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center')

ax = axes[1, 1]
top_features = feature_importance.head(8)
ax.barh(top_features['feature'], top_features['importance'], color='green', alpha=0.7)
ax.set_xlabel('Importance')
ax.set_title('Feature Importance (Permutation)')
ax.grid(True, alpha=0.3)

ax = axes[1, 2]
df['predicted_all'] = svm.predict(scaler.transform(X))
df_clean = df[df['predicted_all'] == 1]
sample_clean = df_clean.iloc[::500]
ax.plot(sample_clean['time_s'], sample_clean['counts'], 'g.', markersize=2, alpha=0.7)
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Photon Count')
ax.set_title('Denoised Data (SVM Filtered)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
png_path = config.RESULTS_GRAPHS_DIR / "svm_spdc_results.png"
pdf_path = config.RESULTS_PDF_DIR / "svm_spdc_results.pdf"
plt.savefig(str(png_path), dpi=300)
plt.savefig(str(pdf_path))
plt.close()

print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")

print("\n[9] Saving results...")

df['ml_prediction'] = df['predicted_all']
pred_csv = config.RESULTS_CSV_DIR / "svm_predictions.csv"
df[['time_s', 'counts', 'ml_prediction']].to_csv(pred_csv, index=False)
print(f"Saved: {pred_csv}")

denoised = df[df['ml_prediction'] == 1][['time_s', 'counts']]
denoised_csv = config.RESULTS_CSV_DIR / "svm_denoised_data.csv"
denoised.to_csv(denoised_csv, index=False)
print(f"Saved: {denoised_csv}")

print("\n" + "="*70)
print("REAL ML COMPLETE")
print("="*70)
print("")
print("RESULTS SUMMARY:")
print("  SVM Accuracy: " + str(round(accuracy, 4)))
print("  ROC AUC: " + str(round(roc_auc, 4)))
print("  Cross-validation mean: " + str(round(cv_scores.mean(), 4)))
print("")
print("Files created:")
print("  - svm_spdc_results.png (6-panel figure)")
print("  - svm_spdc_results.pdf")
print("  - svm_predictions.csv")
print("  - svm_denoised_data.csv")
print("="*70)