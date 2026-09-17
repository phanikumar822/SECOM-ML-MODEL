import os
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

print("Loading AI Model and setting up Interactive Terminal...")
pipe = joblib.load('secom_defect_model.pkl')

# Extract an actual non-defective (passing, label 0) row as template for background 585 sensors
baseline_file = 'secom_baseline.pkl'
if 'baseline_chip' in pipe:
    baseline_chip = pipe['baseline_chip']
elif os.path.exists(baseline_file):
    baseline_chip = joblib.load(baseline_file)
else:
    print("Fetching dataset and extracting an actual non-defective (passing) template...")
    from ucimlrepo import fetch_ucirepo
    secom = fetch_ucirepo(id=179)
    original = secom.data.original
    X = original.drop(columns=['class', 'timestamp']).copy()
    X.columns = [str(i) for i in range(X.shape[1])]
    y = original['class'].map({-1: 0, 1: 1}).values
    
    # Filter for non-defective rows (target label is 0) and use an actual passing row as template
    passing_rows = X[y == 0]
    baseline_chip = passing_rows.iloc[[0]].copy()
    joblib.dump(baseline_chip, baseline_file)

# Get optimal decision threshold from model artifact
optimal_threshold = pipe.get('optimal_threshold', 0.2957)

# Get top 5 most important sensors from saved model
xgb_model = pipe['model']
importances = xgb_model.feature_importances_
selected_features = pipe['selected_features']
top_indices_in_subset = np.argsort(importances)[::-1][:5]
top_original_sensors = [selected_features[i] for i in top_indices_in_subset]

print("\n" + "="*50)
print(" INTERACTIVE SEMICONDUCTOR PREDICTION")
print("="*50)
print("The AI requires 590 sensors. To save time, we have filled")
print("the background 585 sensors using an actual passing (non-defective) chip.")
print("You will only input the values for the 5 most critical sensors.")
print("(Press ENTER to use the baseline passing value)\n")

user_chip = baseline_chip.copy()

for sensor_id in top_original_sensors:
    default_val = baseline_chip[str(sensor_id)].values[0]
    # If the raw sample has NaN for this sensor, use the imputer's median value for clean display
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

print("\n" + "="*50)
print(" RESULTS")
print("="*50)
if prob >= optimal_threshold:
    print("!!! PREDICTION: FAIL (DEFECTIVE) !!!")
else:
    print("*** PREDICTION: PASS (GOOD) ***")
print(f"Failure Probability: {prob:.2%}")
print(f"Decision Threshold:  {optimal_threshold:.2%}")
print("="*50)
