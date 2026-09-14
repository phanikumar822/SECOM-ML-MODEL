import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

print("Loading Data and Model...\n")
# Load model pipeline
pipe = joblib.load('secom_defect_model.pkl')
selected_features = pipe['selected_features']

# Fetch data to recreate the test set
secom = fetch_ucirepo(id=179)
original = secom.data.original
X = original.drop(columns=['class', 'timestamp']).copy()
X.columns = [str(i) for i in range(X.shape[1])]
y = original['class'].map({-1: 0, 1: 1}).values

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Pick 3 random chips from the test set
np.random.seed(101) # Set seed for reproducible random selection
random_indices = np.random.choice(len(X_test), 3, replace=False)

print("==================================================")
print("RANDOM CHIP SAMPLES FROM TEST DATA")
print("==================================================")

for i, idx in enumerate(random_indices):
    chip_data = X_test.iloc[[idx]]
    actual_label = y_test[idx]
    
    # Run Inference
    data_imp = pipe['imputer'].transform(chip_data)
    data_scaled = pipe['scaler'].transform(data_imp)
    data_sel = pipe['selector'].transform(data_scaled)
    
    prediction = pipe['model'].predict(data_sel)[0]
    prob = pipe['model'].predict_proba(data_sel)[0][1]
    
    # We have 590 sensors, which is too many to print. 
    # Let's just print the first 6 sensors that the model selected as "important".
    sensors_to_show = selected_features[:6]
    
    print(f"CHIP SKEW #{i+1} (Test Set Index: {idx})")
    print("--- Raw Sensor Values (Key Sensors) ---")
    for sensor_idx in sensors_to_show:
        val = chip_data[str(sensor_idx)].values[0]
        print(f"  Sensor {sensor_idx:^3}: {val}")
    
    print("--- Results ---")
    print(f"  Actual Status:       {'FAIL (Defective)' if actual_label == 1 else 'PASS (Good)'}")
    print(f"  Model Prediction:    {'FAIL' if prediction == 1 else 'PASS'}")
    print(f"  Failure Probability: {prob:.2%}")
    print("==================================================")
