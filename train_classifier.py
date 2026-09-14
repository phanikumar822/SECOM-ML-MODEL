import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib

print("1. Fetching SECOM dataset...")
secom = fetch_ucirepo(id=179)
original = secom.data.original

# Features and Target
X = original.drop(columns=['class', 'timestamp']).copy()
X.columns = [str(i) for i in range(X.shape[1])]
y = original['class'].map({-1: 0, 1: 1}).values # 0=Pass, 1=Fail

# Train/Test Split (stratified to maintain the 93/7 ratio)
print("2. Splitting data into Train and Test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print("3. Preprocessing: Imputing missing values and scaling...")
# Imputation
imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)

print("4. Feature Selection: Identifying the most important sensors...")
# Use an initial model to find important features
selector_model = xgb.XGBClassifier(n_estimators=50, random_state=42)
selector_model.fit(X_train_scaled, y_train)

# Select features that have importance above the median importance
selector = SelectFromModel(selector_model, prefit=True, max_features=50)
X_train_sel = selector.transform(X_train_scaled)
X_test_sel = selector.transform(X_test_scaled)

selected_indices = selector.get_support(indices=True)
print(f"   Selected {len(selected_indices)} out of {X.shape[1]} features.")

print("5. Handling Class Imbalance: Applying SMOTE...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_sel, y_train)
print(f"   Training set before SMOTE: {np.bincount(y_train)} (Pass/Fail)")
print(f"   Training set after SMOTE:  {np.bincount(y_train_smote)} (Pass/Fail)")

print("6. Training the Final Classification Model...")
# Train final classifier
clf = xgb.XGBClassifier(
    n_estimators=150, 
    learning_rate=0.05, 
    max_depth=4, 
    random_state=42
)
clf.fit(X_train_smote, y_train_smote)

print("\n==========================================")
print("7. EVALUATION ON UNSEEN TEST DATA")
print("==========================================")
y_pred = clf.predict(X_test_sel)
y_pred_proba = clf.predict_proba(X_test_sel)[:, 1]

print(classification_report(y_test, y_pred, target_names=["Pass (0)", "Fail (1)"]))

roc_auc = roc_auc_score(y_test, y_pred_proba)
pr_auc = average_precision_score(y_test, y_pred_proba)
print(f"ROC-AUC Score: {roc_auc:.4f}")
print(f"PR-AUC Score (Precision-Recall): {pr_auc:.4f}")

print("\n8. Saving pipeline for future use...")
pipeline = {
    'imputer': imputer,
    'scaler': scaler,
    'selector': selector,
    'model': clf,
    'selected_features': selected_indices
}
joblib.dump(pipeline, 'secom_defect_model.pkl')
print("   -> Model saved to 'secom_defect_model.pkl'")

print("\n==========================================")
print("HOW TO USE IT (Inference Example)")
print("==========================================")
# Simulate a new chip coming off the manufacturing line
new_chip_sensors = X_test.iloc[[0]].copy() # Take the first row from test set as an example
actual_label = y_test[0]

def predict_new_chip(sensor_data_df):
    """
    Function to use the saved model on new data.
    """
    # Load the saved pipeline components
    pipe = joblib.load('secom_defect_model.pkl')
    
    # Run the raw data through the same pipeline
    data_imp = pipe['imputer'].transform(sensor_data_df)
    data_scaled = pipe['scaler'].transform(data_imp)
    data_sel = pipe['selector'].transform(data_scaled)
    
    # Get prediction
    prediction = pipe['model'].predict(data_sel)[0]
    probability = pipe['model'].predict_proba(data_sel)[0][1]
    
    result = "FAIL (Defective)" if prediction == 1 else "PASS (Good)"
    return result, probability

prediction, prob = predict_new_chip(new_chip_sensors)
print(f"Actual status of chip: {'FAIL' if actual_label == 1 else 'PASS'}")
print(f"Model prediction:      {prediction} (Probability of failure: {prob:.1%})")
print("==========================================")
