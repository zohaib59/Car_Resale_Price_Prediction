! pip install numpy==1.26.4
! pip install pandas==2.2.2
! pip install scipy==1.12.0
! pip install matplotlib==3.9.2
! pip install seaborn==0.13.2
! pip install scikit-learn==1.4.2
! pip install joblib==1.4.2
! pip install xgboost==2.1.3

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Config
DATA_PATH = "used_car1.csv"   # <-- your dataset
TARGET_COL = "estimated_final_price"  # regression target column
TEST_SIZE = 0.2
RANDOM_STATE = 42
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load Data
data = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)
if TARGET_COL not in data.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found")

X = data.drop(columns=[TARGET_COL])
y = data[TARGET_COL].astype(float)

# Column Identification
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns

# Preprocessing Pipelines
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
], n_jobs=-1)

# Models (only 3)
models = {
    "LinearRegression": LinearRegression(n_jobs=-1),
    "RandomForest": RandomForestRegressor(
        n_estimators=200, max_depth=12, n_jobs=-1, random_state=RANDOM_STATE
    ),
    "XGBRegressor": XGBRegressor(
        n_estimators=300, max_depth=8, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE
    )
}

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# Evaluation Loop
results = {}
for name, model in models.items():
    print("=" * 80)
    print(f"MODEL: {name}")

    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    # Metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # MAPE (safe division)
    train_mape = np.mean(np.abs((y_train - y_train_pred) / np.clip(y_train, 1e-10, None))) * 100
    test_mape = np.mean(np.abs((y_test - y_test_pred) / np.clip(y_test, 1e-10, None))) * 100

    results[name] = {
        "train_mae": train_mae, "test_mae": test_mae,
        "train_rmse": train_rmse, "test_rmse": test_rmse,
        "train_r2": train_r2, "test_r2": test_r2,
        "train_mape": train_mape, "test_mape": test_mape
    }

    print(f"[{name}] Train MAE   : {train_mae:.4f}")
    print(f"[{name}] Test  MAE   : {test_mae:.4f}")
    print(f"[{name}] Train RMSE  : {train_rmse:.4f}")
    print(f"[{name}] Test  RMSE  : {test_rmse:.4f}")
    print(f"[{name}] Train R²    : {train_r2:.4f}")
    print(f"[{name}] Test  R²    : {test_r2:.4f}")
    print(f"[{name}] Train MAPE  : {train_mape:.2f}%")
    print(f"[{name}] Test  MAPE  : {test_mape:.2f}%")

    joblib.dump(pipeline, os.path.join(MODEL_DIR, f"{name}.joblib"))































