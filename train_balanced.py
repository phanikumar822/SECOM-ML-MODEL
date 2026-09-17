import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_recall_curve, accuracy_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("--- BUILDING BALANCED F1-OPTIMIZED MODEL ---")
print("Fetching and cleaning data...")
secom = fetch_ucirepo(id=179)
original = secom.data.original
X = original.drop(columns=['class', 'timestamp']).copy()
y = original['class'].map({-1: 0, 1: 1}).values

missing = X.isnull().mean()
X = X.loc[:, missing < 0.4]
X = X.loc[:, X.nunique() > 1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

pipeline = ImbPipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42, k_neighbors=3)),
    ('model', xgb.XGBClassifier(
        n_estimators=150, 
        max_depth=4, 
        learning_rate=0.05, 
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ))
])

print("Finding the perfect mathematical balance (F1-Score)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
probas_cv = cross_val_predict(pipeline, X_train, y_train, cv=cv, method='predict_proba')[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_train, probas_cv)

# Optimize for F1-Score (The perfect balance between Accuracy and catching defects)
f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
best_idx = np.argmax(f1_scores)
if best_idx == len(thresholds):
    best_idx -= 1
best_threshold = thresholds[best_idx]

print(f"Optimal Balanced Threshold: {best_threshold:.2%}\n")

print("Training final model...")
pipeline.fit(X_train, y_train)

probas_test = pipeline.predict_proba(X_test)[:, 1]
final_predictions = (probas_test >= best_threshold).astype(int)

print("==========================================")
print("BALANCED PREDICTION RESULTS")
print("==========================================")
acc = accuracy_score(y_test, final_predictions)
print(f"Overall Balanced Accuracy: {acc:.2%}\n")

print(classification_report(y_test, final_predictions, target_names=["Pass (0)", "Fail (1)"]))
