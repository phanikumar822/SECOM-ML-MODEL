import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("1. Fetching Data...")
secom = fetch_ucirepo(id=179)
original = secom.data.original
X = original.drop(columns=['class', 'timestamp']).copy()
y = original['class'].map({-1: 0, 1: 1}).values

print("2. Advanced Cleaning (Removing Noise)...")
# OPTIMIZATION 1: Drop columns that are mostly empty (>40% missing)
missing_frac = X.isnull().mean()
cols_to_keep = missing_frac[missing_frac < 0.4].index
X = X[cols_to_keep]

print("3. Splitting Data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# OPTIMIZATION 2: Remove constant features (sensors that never change)
var_thresh = VarianceThreshold(threshold=0.0)
X_train_var = var_thresh.fit_transform(X_train)
X_test_var = var_thresh.transform(X_test)

print("4. Imputing and Scaling...")
imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train_var)
X_test_imp = imputer.transform(X_test_var)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)

print("5. Stricter Feature Selection...")
# OPTIMIZATION 3: Stricter feature selection to prevent the model from getting confused
selector_model = xgb.XGBClassifier(n_estimators=100, random_state=42, max_depth=3)
selector_model.fit(X_train_scaled, y_train)
selector = SelectFromModel(selector_model, prefit=True, max_features=30)
X_train_sel = selector.transform(X_train_scaled)
X_test_sel = selector.transform(X_test_scaled)

print("6. SMOTE...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train_sel, y_train)

print("7. Training Optimized Model...")
# OPTIMIZATION 4: Hyperparameter Tuning to prevent overfitting
clf = xgb.XGBClassifier(
    n_estimators=300,        # More trees
    learning_rate=0.01,      # Learn slower
    max_depth=5,
    subsample=0.8,           # Randomly sample data to prevent memorization
    colsample_bytree=0.8,    # Randomly sample features
    random_state=42
)
clf.fit(X_train_smote, y_train_smote)

print("\n==========================================")
print("OPTIMIZED EVALUATION")
print("==========================================")
y_pred = clf.predict(X_test_sel)
y_pred_proba = clf.predict_proba(X_test_sel)[:, 1]

acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_pred_proba)

print(f"Overall Accuracy: {acc:.2%}")
print(f"ROC-AUC Score:    {roc:.4f}\n")

print(classification_report(y_test, y_pred, target_names=["Pass (0)", "Fail (1)"]))
