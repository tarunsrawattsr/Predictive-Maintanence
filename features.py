import pandas as pd

# We use the last 20 cycles to understand how sensors are changing
WINDOW = 20

# Maximum Remaining Useful Life used by our model
RUL_CAP = 125

# We use these 14 sensors because their readings change as the engine gets older
# The remaining sensors are mostly constant in the FD001 dataset so they are not very useful for predicting engine failure
SENSORS = [
    "s2", "s3", "s4", "s7", "s8", "s9", "s11",
    "s12", "s13", "s14", "s15", "s17", "s20", "s21",
]

# Column names used in the NASA C-MAPSS dataset
# Each row contains:
# - Engine number
# - Cycle number
# - 3 operating settings
# - 21 sensor readings
ALL_COLUMNS = (
    ["unit", "cycle", "op1", "op2", "op3"]
    + [f"s{i}" for i in range(1, 22)]
)

def create_features(data):
    """
    Create features for all engines in the training dataset.

    We calculate the last value, mean, standard deviation,
    and trend for each selected sensor.

    The function returns a DataFrame containing:
    - 57 input features
    - RUL (the value we want to predict)
    - Engine ID
    """

    # Store the calculated features of all engines
    all_engines = []

    # Process each engine separately
    for _, engine in data.groupby("unit"):

        # Make sure the engine readings are in cycle order
        engine = engine.sort_values("cycle")

        # Create an empty DataFrame to store the features
        features = pd.DataFrame(index=engine.index)

        # Include the current cycle number as one of the features
        features["cycle"] = engine["cycle"]

        # Calculate features for each of the 14 selected sensors
        for sensor in SENSORS:

            # Get all readings of the current sensor
            values = engine[sensor]

            # Last value:
            # Store the sensor reading from the current cycle
            features[sensor + "_last"] = values

            # compute mean:
            features[sensor + "_mean"] = (
                values.rolling(
                    WINDOW,
                    min_periods=1
                ).mean()
            )

            # compute standard deviation:
            features[sensor + "_std"] = (
                values.rolling(
                    WINDOW,
                    min_periods=1
                ).std().fillna(0)
            )

            # determine trend:
            features[sensor + "_trend"] = (
                values - values.shift(WINDOW - 1)
            ).fillna(0)

        # Add ground truth RUL
        features["rul"] = engine["rul"]

        # Keep the engine ID so we know which engine
        # each feature row belongs to
        features["unit"] = engine["unit"]

        # Add this engine's features to our list
        all_engines.append(features)

    # Combine the features of all engines into one DataFrame
    # Reset the index to give the rows a new numbering
    return pd.concat(all_engines).reset_index(drop=True)


def make_feature_row(history, feature_columns):
    """
    Create one feature row for predicting a machine's RUL.

    This function is used by our prediction API.

    The API receives recent sensor readings from a machine,
    and this function converts them into the same 57 features
    that were used during model training.
    """

    # Keep only the most recent 20 sensor readings
    # Older readings are not needed for our feature calculations
    history = history.tail(WINDOW)

    # Get the latest reading from the machine
    current = history.iloc[-1]

    # Start creating the feature dictionary
    # The current cycle number is our first feature
    features = {
        "cycle": float(current["cycle"])
    }

    # Calculate the same features we used during training
    for sensor in SENSORS:

        # Get the recent readings of this sensor
        values = history[sensor]

        # Last value:
        # Use the most recent sensor reading
        features[sensor + "_last"] = float(
            values.iloc[-1]
        )

        # calculate mean:
        features[sensor + "_mean"] = float(
            values.mean()
        )

        # calculate standard deviation:
        features[sensor + "_std"] = (
            float(values.std()) if len(values) > 1 else 0.0
        )

        # determine trend:
        if len(values) >= WINDOW:
            features[sensor + "_trend"] = float(
                values.iloc[-1] - values.iloc[-WINDOW]
            )

        else:
            # If we have fewer than 20 readings,
            # we cannot calculate the full trend
            # so we use 0 instead
            features[sensor + "_trend"] = 0.0

    # Convert the features into a DataFrame
    # feature_columns makes sure the columns are in the
    # exact same order as they were during training
    return pd.DataFrame(
        [features],
        columns=feature_columns
    )

def get_status(rul):
    """
    Convert the predicted RUL into a machine status.

    We use three categories:
    Critical - Less than 20 cycles remaining
    Warning  - Between 20 and 49 cycles remaining
    Healthy  - 50 or more cycles remaining
    """

    # Machine is close to failure
    if rul < 20:
        return "Critical"

    # Machine still has some life left but needs monitoring
    if rul < 50:
        return "Warning"

    # Machine has enough remaining life
    return "Healthy"

def get_health(rul):
    """
    Convert predicted RUL into a health percentage.

    We use 125 cycles as the maximum value.

    For example:
    RUL = 125 gives 100% health
    RUL = 62.5 gives 50% health
    RUL = 0 gives 0% health

    This percentage is a simple estimate based on RUL,
    not a direct measurement of the engine's condition.
    """

    # Convert remaining useful life into a percentage
    health = (rul / RUL_CAP) * 100

    # Keep the health value between 0 and 100
    # This prevents negative values or percentages above 100
    return max(
        0.0,
        min(100.0, health)
    )