# =========================================================
# BUSINESS MANAGEMENT SYSTEM
# Gross Margin Prediction - Advanced Training Pipeline
# =========================================================

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor


# =========================================================
# 1. DATASET CREATION
# =========================================================

def make_dataset(n=5000):
    """Create dataset with only numeric features used by the model."""

    # ---------- PAPER BAG BUSINESS ----------
    paper = pd.DataFrame({
        "type": 0,  # 0 = paper
        "Quantity": np.random.randint(100, 2000, n),
        "Unit_Price": np.random.uniform(5, 25, n),
    })
    paper["Total_Value"] = paper["Quantity"] * paper["Unit_Price"]

    # ---------- SOLAR BUSINESS ----------
    solar = pd.DataFrame({
        "type": 1,  # 1 = solar
        "Quantity": np.random.randint(1, 100, n),
        "Unit_Price": np.random.uniform(5000, 50000, n),
    })
    solar["Total_Value"] = solar["Quantity"] * solar["Unit_Price"]

    df = pd.concat([paper, solar], ignore_index=True)

    return df


# =========================================================
# 2. FEATURE ENGINEERING
# =========================================================

def add_features(df):
    """Add target variable (gross margin) based on cost/revenue."""

    # Target creation: gross margin = (revenue - cost) / revenue
    cost = df["Total_Value"]
    revenue = cost * np.random.uniform(1.1, 1.6, len(df))

    df["gross_margin"] = (revenue - cost) / revenue

    return df


# =========================================================
# 3. MODEL TRAINING
# =========================================================

def train_model():

    print("\nCreating dataset...")
    df = make_dataset()

    print("Engineering features...")
    df = add_features(df)

    X = df.drop("gross_margin", axis=1)
    y = df["gross_margin"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------- Candidate Models --------
    models = {

        "RandomForest":
            RandomForestRegressor(n_estimators=200, random_state=42),

        "XGBoost":
            XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42
            )
    }

    results = {}
    trained_models = {}

    print("\nTraining models...\n")

    for name, model in models.items():

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)

        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)

        print(f"{name} → R2:{r2:.4f} | MAE:{mae:.4f}")

        results[name] = r2
        trained_models[name] = pipeline

    # -------- Best Model Selection --------
    best_name = max(results, key=results.get)
    best_model = trained_models[best_name]

    print(f"\n✅ BEST MODEL = {best_name}")

    # Save model + feature columns
    joblib.dump({
        "model": best_model,
        "features": list(X.columns)
    }, "model.pkl")

    print("✅ model.pkl saved")


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    train_model()