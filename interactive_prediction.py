import os
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

print("Loading AI Model and setting up Interactive Terminal...")
pipe = joblib.load('secom_defect_model.pkl')

# Load baseline chip and defective profiles from model artifact or local cache
if 'baseline_chip' in pipe and 'defective_chips' in pipe:
    baseline_chip = pipe['baseline_chip']
    defective_chips = pipe['defective_chips']
elif os.path.exists('secom_baseline.pkl'):
    cache = joblib.load('secom_baseline.pkl')
    if isinstance(cache, dict):
        baseline_chip = cache['baseline']
        defective_chips = cache['defective']
    else:
        baseline_chip = cache
        defective_chips = None
else:
    defective_chips = None

if defective_chips is None:
    print("Fetching dataset to extract defective profiles...")
    from ucimlrepo import fetch_ucirepo
    secom = fetch_ucirepo(id=179)
    original = secom.data.original
    X = original.drop(columns=['class', 'timestamp']).copy()
    X.columns = [str(i) for i in range(X.shape[1])]
    y = original['class'].map({-1: 0, 1: 1}).values
    
    baseline_chip = X[y == 0].iloc[[0]].copy()
    defective_chips = X[y == 1].head(3).copy()

# Get optimal decision threshold from model artifact
optimal_threshold = pipe.get('optimal_threshold', 0.2957)

# Get top 5 most important sensors from saved model
xgb_model = pipe['model']
importances = xgb_model.feature_importances_
selected_features = pipe['selected_features']
top_indices_in_subset = np.argsort(importances)[::-1][:5]
top_original_sensors = [selected_features[i] for i in top_indices_in_subset]

# PRINT CLEAN TERMINAL TABLE OF ACTUAL DEFECTIVE ROWS
print("\n" + "="*80)
print(" KNOWN REAL-WORLD DEFECTIVE CHIP PROFILES (TARGET = 1)")
print("="*80)
headers = [f"Sensor {s}" for s in top_original_sensors]
header_str = f"{'Sample':<12} | " + " | ".join([f"{h:^11}" for h in headers])
print(header_str)
print("-" * len(header_str))

for idx, (row_idx, row) in enumerate(defective_chips.iterrows(), start=1):
    row_vals = []
    for s in top_original_sensors:
        val = row[str(s)]
        if pd.isna(val):
            val = pipe['imputer'].statistics_[int(s)]
        row_vals.append(f"{val:^11.4f}")
    vals_str = " | ".join(row_vals)
    print(f"Defect #{idx:<5} | {vals_str}")

print("="*80)
print("Use these real-world defective values to test the model's decision limits!\n")

print("="*80)
print(" INTERACTIVE SEMICONDUCTOR PREDICTION")
print("="*80)
print("The AI requires 590 sensors. To save time, we have filled")
print("the background 585 sensors using an actual passing (non-defective) chip.")
print("You will only input the values for the 5 most critical sensors.")
print("(Press ENTER to use the baseline passing value)\n")

user_chip = baseline_chip.copy()

for sensor_id in top_original_sensors:
    default_val = baseline_chip[str(sensor_id)].values[0]
    if pd.isna(default_val):
        default_val = pipe['imputer'].statistics_[int(sensor_id)]
    
    while True:
        try:
            user_input = input(f"Enter value for Sensor {sensor_id:^3} (Baseline is {default_val:.4f}): ")
            if user_input.strip() == "":
                val = default_val
            else:
                val = float(user_input)
            
            user_chip[str(sensor_id)] = val
            break
        except ValueError:
            print("  [!] Invalid input. Please enter a number.")

print("\nRunning AI Prediction...")
# Run through pipeline
data_imp = pipe['imputer'].transform(user_chip)
data_scaled = pipe['scaler'].transform(data_imp)
data_sel = pipe['selector'].transform(data_scaled)

prob = pipe['model'].predict_proba(data_sel)[0][1]

print("\n" + "="*80)
print(" RESULTS")
print("="*80)
if prob >= optimal_threshold:
    print("!!! PREDICTION: FAIL (DEFECTIVE) !!!")
else:
    print("*** PREDICTION: PASS (GOOD) ***")
print(f"Failure Probability: {prob:.2%}")
print(f"Decision Threshold:  {optimal_threshold:.2%}")
print("="*80)
