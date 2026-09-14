import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_recall_curve
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("--- BUILDING PRODUCTION-LEVEL PIPELINE ---")

# 1. Fetch & Clean
print("Fetching and cleaning real-world data...")
secom = fetch_ucirepo(id=179)
original = secom.data.original
X = original.drop(columns=['class', 'timestamp']).copy()
y = original['class'].map({-1: 0, 1: 1}).values

# Drop columns with > 40% NaNs and zero variance
missing_frac = X.isnull().mean()
X = X.loc[:, missing_frac < 0.4]
X = X.loc[:, X.nunique() > 1] # Drop constant columns

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 3. Impute & Scale
print("Imputing and scaling...")
imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)

# 4. Production XGBoost (Using scale_pos_weight for Cost-Sensitive Learning)
print("Training Cost-Sensitive XGBoost Model...")
# Calculate ratio of negative to positive class
ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)

clf = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=ratio, # Forces AI to care 14x more about defects
    random_state=42
)
clf.fit(X_train_scaled, y_train)

# 5. Find the OPTIMAL Production Threshold
print("Optimizing decision thresholds...")
probas_train = clf.predict_proba(X_train_scaled)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_train, probas_train)

# We want to maximize the F2 score (which heavily penalizes missing a defect)
# Adding 1e-8 to avoid division by zero
f2_scores = (5 * precisions * recalls) / (4 * precisions + recalls + 1e-8)
optimal_idx = np.argmax(f2_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"\n[SYSTEM] Optimal Production Threshold Calculated: {optimal_threshold:.2%}")
print("[SYSTEM] The AI will now automatically flag any chip over this probability as a DEFECT.\n")

# 6. Evaluate on Unseen Test Data using the Custom Threshold
probas_test = clf.predict_proba(X_test_scaled)[:, 1]

# Generate final production predictions based on the optimized threshold
production_predictions = (probas_test >= optimal_threshold).astype(int)

print("==========================================")
print("PRODUCTION-LEVEL PREDICTION RESULTS")
print("==========================================")
print(classification_report(y_test, production_predictions, target_names=["Pass (0)", "Fail (1)"]))
