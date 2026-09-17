import os
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

print("Loading AI Model and setting up Interactive Terminal...")
pipe = joblib.load('secom_defect_model.pkl')

# Load baseline chip from local disk cache if available (instant load)
baseline_file = 'secom_baseline.pkl'
if os.path.exists(baseline_file):
    baseline_chip = joblib.load(baseline_file)
else:
    print("Fetching initial baseline dataset from repository (one-time setup)...")
    from ucimlrepo import fetch_ucirepo
    secom = fetch_ucirepo(id=179)
    original = secom.data.original
    X = original.drop(columns=['class', 'timestamp']).copy()
    X.columns = [str(i) for i in range(X.shape[1])]
    baseline_chip = X.median().to_frame().T
    joblib.dump(baseline_chip, baseline_file)

# Get the top 5 most important sensors from the saved model
xgb_model = pipe['model']
importances = xgb_model.feature_importances_
selected_features = pipe['selected_features']
top_indices_in_subset = np.argsort(importances)[::-1][:5]
top_original_sensors = [selected_features[i] for i in top_indices_in_subset]

print("\n" + "="*50)
print(" INTERACTIVE SEMICONDUCTOR PREDICTION")
print("="*50)
print("The AI requires 590 sensors. To save time, we have filled")
print("the background sensors with 'average' healthy factory values.")
print("You will only input the values for the 5 most critical sensors.")
print("(Press ENTER to just use the average value)\n")

user_chip = baseline_chip.copy()

for sensor_id in top_original_sensors:
    default_val = baseline_chip[str(sensor_id)].values[0]
    
    while True:
        try:
            user_input = input(f"Enter value for Sensor {sensor_id:^3} (Average is {default_val:.4f}): ")
            if user_input.strip() == "":
                val = default_val
            else:
                val = float(user_input)
            
            user_chip[str(sensor_id)] = val
            break
        except ValueError:
            print("  [!] Invalid input. Please enter a number.")

print("\nRunning AI Prediction...")
# Run the user's custom chip through the saved machine learning pipeline
data_imp = pipe['imputer'].transform(user_chip)
data_scaled = pipe['scaler'].transform(data_imp)
data_sel = pipe['selector'].transform(data_scaled)

prediction = pipe['model'].predict(data_sel)[0]
prob = pipe['model'].predict_proba(data_sel)[0][1]

print("\n" + "="*50)
print(" RESULTS")
print("="*50)
# Use the production threshold of 23.48% instead of the default 50%
if prob >= 0.2348:
    print(f"!!! PREDICTION: FAIL (DEFECTIVE) !!!")
else:
    print(f"*** PREDICTION: PASS (GOOD) ***")
print(f"Failure Probability: {prob:.2%}")
print("="*50)
