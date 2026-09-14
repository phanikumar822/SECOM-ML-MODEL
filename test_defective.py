import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

print("Fetching SECOM dataset...")
secom = fetch_ucirepo(id=179)
original = secom.data.original

# Recreate the exact same data split
X = original.drop(columns=['class', 'timestamp']).copy()
X.columns = [str(i) for i in range(X.shape[1])]
y = original['class'].map({-1: 0, 1: 1}).values

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def predict_new_chip(sensor_data_df):
    # Load the saved pipeline
    pipe = joblib.load('secom_defect_model.pkl')
    
    # Process
    data_imp = pipe['imputer'].transform(sensor_data_df)
    data_scaled = pipe['scaler'].transform(data_imp)
    data_sel = pipe['selector'].transform(data_scaled)
    
    # Predict
    prediction = pipe['model'].predict(data_sel)[0]
    probability = pipe['model'].predict_proba(data_sel)[0][1]
    
    result = "FAIL (Defective)" if prediction == 1 else "PASS (Good)"
    return result, probability

# Find indices where the actual status is FAIL (1)
defective_indices = np.where(y_test == 1)[0]

print(f"\nFound {len(defective_indices)} defective chips in the unseen test set.")
print("Testing the model on the first 5 defective chips...\n")
print("==================================================")

for i in range(min(5, len(defective_indices))):
    idx = defective_indices[i]
    defective_chip_sensors = X_test.iloc[[idx]].copy()
    
    prediction, prob = predict_new_chip(defective_chip_sensors)
    
    print(f"Chip Test #{i+1}")
    print(f"Actual status:      FAIL (Defective)")
    print(f"Model prediction:   {prediction}")
    print(f"Failure Probability: {prob:.1%}")
    print("--------------------------------------------------")
