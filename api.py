import pickle
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from features import SENSORS, get_health, get_status, make_feature_row

# path to model
MODEL_FILE = Path("model.pkl")

# create FastAPI application
app = FastAPI(title="Predictive Maintenance API")


# Check if the trained model file exists
if MODEL_FILE.exists():

    # Open the saved model file
    # We use pickle because train.py saves the model in this format
    with open(MODEL_FILE, "rb") as file:
        saved_model = pickle.load(file)

    # Get the trained LightGBM model
    MODEL = saved_model["model"]

    # Get the feature names used during training
    FEATURE_COLUMNS = saved_model["feature_columns"]

else:

    MODEL = None
    FEATURE_COLUMNS = []


class PredictRequest(BaseModel):
    """
    Define the data that the simulator sends to the API.

    The request contains recent sensor readings of one machine.

    Example:
    {
        "history": [
            {"cycle": 41, "s2": 642.1, "s3": 1580.5},
            {"cycle": 42, "s2": 642.4, "s3": 1581.0}
        ]
    }

    In the actual request, each reading must contain the
    cycle number and all 14 sensors used by our model.

    We use up to the latest 20 readings for prediction.
    """

    # Store the sensor history as a list of dictionaries
    # Each dictionary represents one machine cycle
    history: list[dict[str, float]]


class PredictResponse(BaseModel):
    """
    Define the values returned by the prediction API.

    RUL tells us the estimated remaining life of the machine.
    Health is a percentage calculated from the predicted RUL.
    Status tells us whether the machine is healthy or needs attention.
    """

    # Predicted number of cycles remaining before failure
    rul: float

    # Machine health percentage between 0 and 100
    health: float

    # Machine status: Healthy, Warning, or Critical
    status: str


@app.get("/health")
def health_check():
    """
    Check whether the API is running.

    This endpoint also tells us if the trained model
    was loaded successfully.
    """

    # Return the API status and whether the model is available
    return {
        "status": "ok",
        "model_loaded": MODEL is not None
    }

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Predict the Remaining Useful Life of a machine.

    The simulator sends recent sensor readings to this endpoint.
    We prepare the features, use the trained model for prediction,
    and return the machine's RUL, health, and status.
    """

    # Check if the trained model is available
    # If not, return an error asking the user to train it first
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please run train.py first.",
        )

    # Check if the request contains at least one sensor reading
    # We cannot predict anything if the history is empty
    if not request.history:
        raise HTTPException(
            status_code=400,
            detail="Please send at least one reading in 'history'.",
        )

    # Convert the received JSON readings into a pandas DataFrame
    # This makes it easier to calculate the required features
    history = pd.DataFrame(request.history)

    # List of columns needed by our machine learning model
    # These include the cycle number and 14 selected sensors
    needed_columns = ["cycle"] + SENSORS

    # Check whether any required sensor readings are missing
    missing = [
        c for c in needed_columns
        if c not in history.columns
    ]

    # Return an error if the request does not contain all sensors
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Readings are missing these values: {missing}",
        )

    # Convert the sensor history into the same 57 features
    # that were used when training the LightGBM model
    features = make_feature_row(
        history,
        FEATURE_COLUMNS
    )

    # Use the trained model to predict RUL
    # The model returns an array, so we take the first value
    predicted_rul = float(
        MODEL.predict(features)[0]
    )

    # Keep RUL between 0 and 125 cycles
    predicted_rul = max(
        0.0,
        min(125.0, predicted_rul)
    )

    # Convert predicted RUL into a health percentage and status and save in DB
    return PredictResponse(
        rul=round(predicted_rul, 1),
        health=round(get_health(predicted_rul), 1),
        status=get_status(predicted_rul),
    )