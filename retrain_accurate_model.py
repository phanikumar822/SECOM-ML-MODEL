import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, precision_recall_curve
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

print("1. Loading SECOM dataset...")
secom = fetch_ucirepo(id=179)
original = secom.data.original

X = original.drop(columns=['class', 'timestamp']).copy()
X.columns = [str(i) for i in range(X.shape[1])]
y = original['class'].map({-1: 0, 1: 1}).values # 0=Pass, 1=Fail

# Extract an actual passing row (label 0) as the baseline template for hidden background sensors
passing_chips = X[y == 0]
baseline_chip = passing_chips.iloc[[0]].copy()

print("2. Splitting data into Train and Test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print("3. Preprocessing: Imputation & Scaling...")
imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)

print("4. Feature Selection...")
pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

selector_model = xgb.XGBClassifier(n_estimators=50, max_depth=3, scale_pos_weight=pos_weight, random_state=42)
selector_model.fit(X_train_scaled, y_train)

selector = SelectFromModel(selector_model, prefit=True, max_features=50)
X_train_sel = selector.transform(X_train_scaled)
X_test_sel = selector.transform(X_test_scaled)

selected_indices = selector.get_support(indices=True)
print(f"   Selected top {len(selected_indices)} sensors.")

print("5. Training XGBoost Classifier...")
clf = xgb.XGBClassifier(
    n_estimators=100, 
    learning_rate=0.03, 
    max_depth=3, 
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=pos_weight,
    random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
probas_cv = cross_val_predict(clf, X_train_sel, y_train, cv=cv, method='predict_proba')[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_train, probas_cv)
f2_scores = (5 * precisions * recalls) / (4 * precisions + recalls + 1e-8)
best_idx = np.argmax(f2_scores)
if best_idx == len(thresholds):
    best_idx -= 1
best_threshold = float(thresholds[best_idx])

clf.fit(X_train_sel, y_train)

print(f"\n[SYSTEM] Calculated Optimal Production Threshold: {best_threshold:.2%}")

probas_test = clf.predict_proba(X_test_sel)[:, 1]
preds_test = (probas_test >= best_threshold).astype(int)

print("\n==========================================")
print("ACCURATE MODEL EVALUATION ON UNSEEN TEST DATA")
print("==========================================")
print(classification_report(y_test, preds_test, target_names=["Pass (0)", "Fail (1)"]))

# Verify Actual Passing Row Baseline Chip Prediction
b_imp = imputer.transform(baseline_chip)
b_scaled = scaler.transform(b_imp)
b_sel = selector.transform(b_scaled)
b_prob = float(clf.predict_proba(b_sel)[0][1])

print("------------------------------------------")
print(f"VERIFICATION - Actual Passing Chip Failure Prob: {b_prob:.2%}")
if b_prob < best_threshold:
    print("VERIFICATION SUCCESSFUL: Actual passing chip correctly predicts PASS (GOOD)!")
else:
    print("WARNING: Baseline chip failure prob above threshold!")

print("------------------------------------------")

pipeline = {
    'imputer': imputer,
    'scaler': scaler,
    'selector': selector,
    'model': clf,
    'selected_features': selected_indices,
    'optimal_threshold': best_threshold,
    'baseline_chip': baseline_chip
}
joblib.dump(pipeline, 'secom_defect_model.pkl')
joblib.dump(baseline_chip, 'secom_baseline.pkl')
print("Model pipeline updated and saved with actual passing row baseline template!")
