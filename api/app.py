import os
import joblib
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

app = FastAPI(
    title="Iris Classification API",
    description="API for predicting Iris flower species using a Decision Tree model.",
    version="1.0.0"
)

# Feature validation limits provided by the UI team (0.8x to 1.2x observed min/max)
FEATURE_LIMITS = {
    "sepal_length": (3.44, 9.48),
    "sepal_width": (1.60, 5.28),
    "petal_length": (0.80, 8.28),
    "petal_width": (0.08, 3.00),
}

class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., description="Sepal length in cm")
    sepal_width: float = Field(..., description="Sepal width in cm")
    petal_length: float = Field(..., description="Petal length in cm")
    petal_width: float = Field(..., description="Petal width in cm")

    def validate_ranges(self):
        """Check if incoming values fall within the agreed-upon safe feature limits."""
        features = {
            "sepal_length": self.sepal_length,
            "sepal_width": self.sepal_width,
            "petal_length": self.petal_length,
            "petal_width": self.petal_width,
        }
        for name, value in features.items():
            min_val, max_val = FEATURE_LIMITS[name]
            if not (min_val <= value <= max_val):
                raise ValueError(
                    f"Feature '{name}' value {value} is out of bounds. "
                    f"Must be between {min_val} and {max_val} cm."
                )

# Class mapping agreed upon by the team
CLASS_MAPPING = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}

# Load local model if available (Day 3+), otherwise fallback to mock prediction
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print(f"Loaded model successfully from {MODEL_PATH}")
        except Exception as e:
            print(f"Failed to load model: {e}")
    else:
            print(f"Warning: Model file not found at {MODEL_PATH}. Running in mock prediction mode.")

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Health check endpoint to verify service status."""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict", status_code=status.HTTP_200_OK)
def predict(payload: IrisFeatures):
    """Receive flower features, validate limits, and return predicted Iris class."""
    try:
        # Validate against Marc's approved UI limits
        payload.validate_ranges()
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

    # Format features into array shape expected by scikit-learn
    features_array = [[
        payload.sepal_length,
        payload.sepal_width,
        payload.petal_length,
        payload.petal_width
    ]]

    if model is not None:
        try:
            prediction_idx = int(model.predict(features_array)[0])
            class_name = CLASS_MAPPING.get(prediction_idx, "unknown")
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction error: {str(e)}"
            )
    else:
        # Mock prediction fallback if model.pkl isn't wired up yet (Day 2 goal)
        prediction_idx = 0
        class_name = CLASS_MAPPING[prediction_idx]

    return {
        "class_id": prediction_idx,
        "class_name": class_name
    }

