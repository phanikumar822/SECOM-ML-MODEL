import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

# ==========================================
# 1. LOAD AND PREPARE DATA
# ==========================================
print("Fetching SECOM dataset from UCI...")
secom = fetch_ucirepo(id=179)

# Load features and targets
# Note: UCI ML repo structure differs slightly from the report.
# Features and targets are None, so we extract them from original.
original = secom.data.original
X = original.drop(columns=['class', 'timestamp']).copy()
# Rename columns to "0", "1", ... to match the PDF's feature indexing
X.columns = [str(i) for i in range(X.shape[1])]
y = original[['class']].copy()

# Convert target:
# -1 = Pass -> 0
#  1 = Fail -> 1
# Note: Using .iloc[:, 0] to get the first column safely in case the column name differs
target_col = y.columns[0]
y.loc[:, target_col] = y[target_col].map({
    -1: 0,
    1: 1
})

# ==========================================
# 2. HANDLE MISSING VALUES
# ==========================================
print("Imputing missing sensor values with median...")
imputer = SimpleImputer(strategy="median")
X_clean = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns
)

# ==========================================
# 3. CONSTRUCT SIMULATED LOTS
# ==========================================
# Extract test timestamps
timestamps = secom.data.original["timestamp"]
X_clean["timestamp"] = pd.to_datetime(timestamps, format="mixed")

# For the prototype, chips tested on
# the same date are considered one lot.
X_clean["lot_id"] = X_clean["timestamp"].dt.date

# Attach target labels
X_clean["target_class"] = y[target_col].values

# ==========================================
# 4. MODULE A:
#    LOT-AWARE DYNAMIC OUTLIER DETECTION
# ==========================================
def calculate_modified_z_score(group, feature_col):
    """
    Calculate Modified Z-Scores using
    the Median Absolute Deviation (MAD)
    within a manufacturing lot.
    """
    median = group[feature_col].median()
    mad = np.median(
        np.abs(group[feature_col] - median)
    )
    
    # Avoid division by zero
    if mad == 0:
        return pd.Series(
            0,
            index=group.index
        )
        
    modified_z = (
        0.6745 * 
        (group[feature_col] - median) / 
        mad
    )
    return modified_z

# Select sensor to analyze
feature_to_test = "0"

print(
    f"Running Module A: "
    f"Lot-aware screening on Sensor "
    f"'{feature_to_test}'..."
)

# Calculate lot-level outlier scores
X_clean["outlier_score"] = (
    X_clean
    .groupby("lot_id", group_keys=False)
    .apply(
        lambda group: 
        calculate_modified_z_score(
            group, 
            feature_to_test
        )
    )
)

# Define Modified Z-Score threshold
outlier_threshold = 3.5

X_clean["flag_module_a"] = (
    X_clean["outlier_score"].abs() 
    > outlier_threshold
)

print(
    f"Module A complete! "
    f"Flagged "
    f"{X_clean['flag_module_a'].sum()} "
    f"chips as Dynamic Lot Outliers."
)

# ==========================================
# 5. MODULE B:
#    TIME-SERIES DRIFT PREDICTOR
# ==========================================
print("\nTraining Module B: Drift Predictor...")

# Simulated early-stage sensor measurements
early_features = [
    str(i) 
    for i in range(10)
]

# Simulated late-stage target parameter
target_feature = "20"

X_drift = X_clean[early_features]
y_drift = X_clean[target_feature]

# ==========================================
# 6. TRAIN / TEST SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X_drift,
    y_drift,
    test_size=0.2,
    random_state=42
)

# ==========================================
# 7. XGBOOST REGRESSION MODEL
# ==========================================
model_b = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.08,
    max_depth=5,
    random_state=42
)

model_b.fit(
    X_train,
    y_train
)

# ==========================================
# 8. MODEL EVALUATION
# ==========================================
predictions = model_b.predict(X_test)

mae = mean_absolute_error(
    y_test,
    predictions
)

print(
    f"Module B complete! "
    f"Drift Prediction MAE: {mae:.4f}"
)

# ==========================================
# 9. PREDICT LATE-STAGE VALUES
# ==========================================
X_clean["predicted_late_value"] = (
    model_b.predict(X_drift)
)

# ==========================================
# 10. LOT-SPECIFIC DRIFT THRESHOLD
# ==========================================
lot_target_medians = (
    X_clean
    .groupby("lot_id")[target_feature]
    .transform("median")
)

# Flag chips where predicted late-stage
# value is significantly above the
# normal lot-level median.
X_clean["flag_module_b"] = (
    X_clean["predicted_late_value"] 
    > (lot_target_medians * 1.5)
)

print(
    f"Module B complete! "
    f"Flagged "
    f"{X_clean['flag_module_b'].sum()} "
    f"chips for high drift risk."
)
