import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import json
import os

RUN_ID = "2026-03-13_run1"
BASE_OUT_DIR = f"artifacts/gemini/{RUN_ID}/taskE_bug_leakage_debug"
CLEAN_DATA = f"artifacts/gemini/{RUN_ID}/taskA_data_audit/outputs/cleaned.csv"

os.makedirs(f"{BASE_OUT_DIR}/outputs", exist_ok=True)
os.makedirs(f"{BASE_OUT_DIR}/plots", exist_ok=True)

df = pd.read_csv(CLEAN_DATA)

# Bug 1: IDs used as predictors
X = df.drop(columns=['fraud_label'])
y = df['fraud_label']

# Bug 2 & 3: non-stratified split, no fixed seed
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Bug 4: preprocessing fit on full data (imputation before split)
X_filled = X_train.fillna(-999) 

# Bug 5: evaluate on training set (or part of it)
model = RandomForestClassifier(n_estimators=10)
X_train_num = X_filled.select_dtypes(include=['number'])
model.fit(X_train_num, y_train)

y_pred = model.predict(X_train_num) # evaluate on training set!
acc = accuracy_score(y_train, y_pred)

metrics = {'Accuracy_Train': acc}
with open(f"{BASE_OUT_DIR}/outputs/metrics_broken.json", "w") as f:
    json.dump(metrics, f, indent=4)
print("Broken pipeline executed.")
