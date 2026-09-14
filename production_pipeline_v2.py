import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_recall_curve
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("--- BUILDING TRUE PRODUCTION PIPELINE (V2) ---")
print("Fetching and cleaning real-world data...")
secom = fetch_ucirepo(id=179)
original = secom.data.original
X = original.drop(columns=['class', 'timestamp']).copy()
y = original['class'].map({-1: 0, 1: 1}).values

# Drop noisy columns
missing = X.isnull().mean()
X = X.loc[:, missing < 0.4]
X = X.loc[:, X.nunique() > 1]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Create a strict pipeline
pipeline = ImbPipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('model', xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=3, 
        learning_rate=0.05, 
        random_state=42
    ))
])

print("Simulating real-world conditions via Cross-Validation...")
# PRO-LEVEL FIX: We use Cross-Validation to generate realistic probabilities,
# preventing the AI from cheating by memorizing the training data.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
probas_cv = cross_val_predict(pipeline, X_train, y_train, cv=cv, method='predict_proba')[:, 1]

print("Calculating mathematically optimal decision threshold...")
# Find the optimal threshold based on REALISTIC cross-val probabilities
precisions, recalls, thresholds = precision_recall_curve(y_train, probas_cv)

# We maximize the F2 score (which heavily penalizes missing a defect)
f2_scores = (5 * precisions * recalls) / (4 * precisions + recalls + 1e-8)
best_idx = np.argmax(f2_scores)
# Safety check to avoid picking a threshold of 100% or 0%
if best_idx == len(thresholds):
    best_idx -= 1
best_threshold = thresholds[best_idx]

print(f"\n[SYSTEM] Optimal Production Threshold Calculated: {best_threshold:.2%}")
print("[SYSTEM] The AI will now automatically flag any chip over this probability as a DEFECT.\n")

print("Training final model and testing on Unseen Data...")
# Train final pipeline on ALL training data
pipeline.fit(X_train, y_train)

# Predict on Unseen Test Set
probas_test = pipeline.predict_proba(X_test)[:, 1]
final_predictions = (probas_test >= best_threshold).astype(int)

print("==========================================")
print("PRODUCTION-LEVEL PREDICTION RESULTS")
print("==========================================")
print(classification_report(y_test, final_predictions, target_names=["Pass (0)", "Fail (1)"]))
