import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("--- BUILDING HIGH-ACCURACY MODEL ---")
print("Fetching and cleaning data...")
secom = fetch_ucirepo(id=179)
original = secom.data.original
X = original.drop(columns=['class', 'timestamp']).copy()
y = original['class'].map({-1: 0, 1: 1}).values

# Clean data
missing_frac = X.isnull().mean()
X = X.loc[:, missing_frac < 0.4]
X = X.loc[:, X.nunique() > 1] 

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Impute & Scale
imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)

print("Training standard AI to maximize pure Accuracy...")
# We remove SMOTE and scale_pos_weight. 
# We just train a standard AI to get the highest accuracy possible.
clf = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)
clf.fit(X_train_scaled, y_train)

# Predict using the default 50% threshold
y_pred = clf.predict(X_test_scaled)

print("\n==========================================")
print("HIGH ACCURACY PREDICTION RESULTS")
print("==========================================")
acc = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy: {acc:.2%}\n")

print(classification_report(y_test, y_pred, target_names=["Pass (0)", "Fail (1)"]))
