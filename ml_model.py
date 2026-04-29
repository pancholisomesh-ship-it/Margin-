import joblib
import numpy as np
import logging
from pathlib import Path
from xgboost import XGBRegressor

# --------------------------------------------------
# Logger
# --------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# Model Path
# --------------------------------------------------
MODEL_PATH = Path("model.pkl")

# ONLY NUMERIC FEATURES FOR ML
FEATURE_NAMES = [
    "type",
    "Quantity",
    "Unit_Price",
    "Total_Value"
]

EXPECTED_FEATURES = len(FEATURE_NAMES)

MARGIN_MIN = -100
MARGIN_MAX = 100

# --------------------------------------------------
# Load Model
# --------------------------------------------------
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "model.pkl not found.\n"
            "Train your model first."
        )

    loaded = joblib.load(MODEL_PATH)

    # Handle both dict format {"model": ..., "features": ...} and direct model
    if isinstance(loaded, dict) and "model" in loaded:
        model = loaded["model"]
        logger.info(f"Model loaded with features: {loaded.get('features', [])}")
    else:
        model = loaded

    logger.info("Model loaded successfully")

    return model


try:
    _model = load_model()
except Exception as e:
    logger.error(e)
    _model = None


# --------------------------------------------------
# Prediction API
# --------------------------------------------------
def predict_margin(input_data):

    _check_model()

    features = _validate(input_data)

    prediction = _model.predict(features)[0]

    prediction = float(prediction)

    # Clamp output
    prediction = max(MARGIN_MIN,
                     min(MARGIN_MAX, prediction))

    result = round(prediction, 2)

    logger.info("Predicted Margin: %.2f%%", result)

    return result


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def _check_model():
    if _model is None:
        raise RuntimeError("Model not loaded.")


def _validate(input_data):

    try:
        arr = np.array(input_data, dtype=float).reshape(1,-1)
    except Exception:
        raise ValueError("Input must be numeric values")

    if arr.shape[1] != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_FEATURES} features "
            f"{FEATURE_NAMES}"
        )

    if not np.isfinite(arr).all():
        raise ValueError("Invalid numeric values detected")

    return arr


# --------------------------------------------------
# Model Info
# --------------------------------------------------
def get_model_info():

    return {
        "model_loaded": _model is not None,
        "features": FEATURE_NAMES,
        "total_features": EXPECTED_FEATURES
    }
   