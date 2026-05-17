# =========================================================
# BUSINESS MANAGEMENT SYSTEM
# Gross Margin Prediction - Advanced Training Pipeline
# =========================================================

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor


# =========================================================
# 1. DATASET CREATION
# =========================================================

def normalize_columns(df):
    def clean_col(col):
        label = col.strip()
        label = label.replace(" (?)", "")
        label = label.replace("?", "")
        label = label.replace("(", "")
        label = label.replace(")", "")
        return "_".join(label.split())
    df = df.rename(columns={col: clean_col(col) for col in df.columns})
    return df


def load_csv_dataset():
    base_path = Path(__file__).parent
    csv_files = [
        (base_path / "paperbags_datasheet.csv", 0),
        (base_path / "solar_material_datasheet.csv", 1)
    ]
    frames = []

    for path, business_type in csv_files:
        if not path.exists():
            continue

        raw = pd.read_csv(path)
        raw = normalize_columns(raw)

        if "Quantity" not in raw.columns or "Unit_Price" not in raw.columns or "Total_Value" not in raw.columns:
            continue

        df = raw[["Quantity", "Unit_Price", "Total_Value"]].copy()
        df["type"] = business_type
        frames.append(df)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        return combined

    return None


def make_dataset(n=5000):
    """Create dataset with only numeric features used by the model."""

    csv_df = load_csv_dataset()
    if csv_df is not None and len(csv_df) >= 50:
        return csv_df

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
    """Add target variable (gross margin) based on business type and features."""
    
    margins = []
    for idx, row in df.iterrows():
        business_type = row["type"]
        quantity = row["Quantity"]
        unit_price = row["Unit_Price"]
        total_value = row["Total_Value"]
        
        # Create realistic margins based on business type and feature interactions
        if business_type == 0:  # Paper bags
            # Paper bags: margin depends on quantity and unit price
            # Higher quantity/price = better margins due to economies of scale
            base_margin = 15 + (np.log1p(quantity) * 3) + (unit_price / 10)
            variance = np.random.uniform(-5, 10)
            margin = max(8, min(50, base_margin + variance))
            
        else:  # Solar (type == 1)
            # Solar: margin depends on system capacity (quantity) and unit cost
            # Higher capacity with reasonable costs = better margins
            base_margin = 18 + (np.log1p(quantity) * 4) - (unit_price / 5000)
            variance = np.random.uniform(-5, 12)
            margin = max(10, min(60, base_margin + variance))
        
        margins.append(margin)
    
    df["gross_margin"] = margins
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

        results[name] = {
            "r2": r2,
            "mae": mae
        }
        trained_models[name] = pipeline

    # -------- Best Model Selection --------
    best_name = max(results, key=lambda k: results[k]["r2"])
    best_model = trained_models[best_name]
    best_metrics = results[best_name]

    print(f"\n✅ BEST MODEL = {best_name}")
    print(f"✅ Best R2 = {best_metrics['r2']:.4f}, MAE = {best_metrics['mae']:.4f}")

    # Save model + feature columns + training metrics
    joblib.dump({
        "model": best_model,
        "features": list(X.columns),
        "metrics": {
            "best_model": best_name,
            "best_r2": best_metrics["r2"],
            "best_mae": best_metrics["mae"],
            "all_metrics": results
        }
    }, "model.pkl")

    print("✅ model.pkl saved")


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    train_model()