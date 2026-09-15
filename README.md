# SECOM Predictive Defect Analysis

This repository contains an end-to-end machine learning pipeline for detecting rare defects in semiconductor manufacturing. It uses the **UCI SECOM Dataset**, which contains 1,567 chips and 590 anonymized sensor readings per chip.

Because semiconductor defects are extremely rare (~6.6% of the dataset), traditional machine learning models that optimize for "Accuracy" fail (they blindly guess "PASS" every time). This project solves that problem by implementing **Cost-Sensitive Learning, SMOTE, and Custom Decision Thresholding** to aggressively hunt for defects.

## 🚀 Key Features & Techniques
* **Automated Data Cleaning:** Automatically removes broken sensors (zero-variance) and sensors missing >40% of their data.
* **Imbalance Handling (SMOTE):** Generates synthetic failure patterns to balance the training data.
* **Cross-Validation:** Uses Stratified 5-Fold Cross-Validation to prevent the model from memorizing the data.
* **Mathematical Threshold Optimization:** Instead of using a default 50% probability to flag a defect, the production pipeline calculates the mathematically perfect threshold (based on the F2-score) to maximize the recall of defective chips while minimizing false alarms.

## 📂 Project Structure

* `production_pipeline_v2.py` 
  **The final production-grade pipeline.** It cleans the data, uses SMOTE and XGBoost, applies cross-validation, and automatically calculates the optimal decision threshold (usually ~23%) to catch nearly 50% of all real-world defects.
* `train_optimized.py` 
  An optimized training script focusing on strict feature selection (top 30 sensors) and XGBoost hyperparameter tuning.
* `train_classifier.py` 
  The baseline classification script that introduces SMOTE and saves the trained model to a `.pkl` file.
* `test_defective.py` 
  An inference testing script that deliberately seeks out known defective chips in the test set to evaluate how the model scores them.
* `predict_random_samples.py`
  Simulates a real-time factory environment by randomly pulling chips, printing their raw sensor measurements, and outputting the AI's PASS/FAIL prediction.

## 🛠️ Installation & Usage

1. **Install Requirements:**
   Make sure you have Python installed, then install the necessary libraries:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Production Pipeline:**
   This script will dynamically download the latest dataset from UCI, clean it, train the model, and output the final production evaluation metrics:
   ```bash
   python production_pipeline_v2.py
   ```

## 🧠 Model Specifications
* **Algorithm:** `XGBoost Classifier`
* **Estimators:** `100`
* **Learning Rate:** `0.05`
* **Max Depth:** `3`
* **Optimization Metric:** `F2-Score` (Prioritizes Recall)
