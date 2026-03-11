import io
import json
import pickle
import urllib.request
import zipfile
from pathlib import Path

# Libraries for training the model and working with data
import lightgbm as lgb
import numpy as np
import pandas as pd

# Importing the feature settings and feature creation function
from features import ALL_COLUMNS, RUL_CAP, create_features


# Paths for the dataset, trained model, and results
DATA_FOLDER = Path("data")
TRAIN_FILE = DATA_FOLDER / "train_FD001.txt"
MODEL_FILE = Path("model.pkl")
RESULTS_FOLDER = Path("results")

# Dataset download links
# If the first link doesn't work, we try the second one
DOWNLOAD_URLS = [
    "https://codeload.github.com/hankroark/Turbofan-Engine-Degradation/zip/refs/heads/master",
    "https://data.nasa.gov/download/ff5v-kuh6/application%2Fzip",
]


def download_data():
    """Download the NASA C-MAPSS dataset if we don't already have it."""

    # No need to download the dataset every time we run the program
    if TRAIN_FILE.exists():
        print("Dataset already exists. Skipping download.")
        return

    # Create the data folder if it doesn't exist
    DATA_FOLDER.mkdir(parents=True, exist_ok=True)
    print("Downloading the NASA C-MAPSS dataset...")

    # Try downloading from the available sources
    for url in DOWNLOAD_URLS:
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                zip_bytes = response.read()

            # The downloaded dataset is a zip file
            # We only need the text files containing engine data
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zip_file:
                for member in zip_file.namelist():
                    file_name = Path(member).name

                    if file_name.endswith(".txt") and "FD00" in file_name:
                        (DATA_FOLDER / file_name).write_bytes(
                            zip_file.read(member)
                        )

            # Check if the training file was downloaded successfully
            if TRAIN_FILE.exists():
                print("Dataset downloaded to the data/ folder.")
                return

        except Exception as error:
            # If one source fails, the program tries the next one
            print(f"Could not download from one source: {error}")

    # Stop the program if we couldn't download the data
    raise RuntimeError("Dataset download failed. Check your internet connection.")


def load_data():
    """Read the engine training data into a pandas DataFrame."""

    # The dataset uses spaces between values instead of commas
    data = pd.read_csv(
        TRAIN_FILE,
        sep=r"\s+",
        header=None,
        names=ALL_COLUMNS,
    )

    # Sort by engine number and cycle so the readings are in order
    return data.sort_values(["unit", "cycle"]).reset_index(drop=True)


def add_rul(data):
    """
    Calculate Remaining Useful Life (RUL) for each engine.

    RUL tells us how many cycles an engine has left before failure.

    For example, if an engine fails at cycle 200:
    At cycle 150, RUL = 50
    At cycle 200, RUL = 0
    """

    # Find the last cycle of each engine
    # In this dataset, the last cycle represents engine failure
    last_cycle = data.groupby("unit")["cycle"].transform("max")

    # Remaining life = last cycle - current cycle
    data["rul"] = last_cycle - data["cycle"]

    # Limit RUL to 125 cycles
    # This is because it's difficult to predict exact remaining life
    # when the engine is still healthy and far from failure
    data["rul"] = data["rul"].clip(upper=RUL_CAP)

    return data


def rmse_score(true_values, predicted_values):
    """
    Calculate Root Mean Squared Error (RMSE).

    This helps us measure how far our predictions are
    from the actual remaining life of the engines.

    A lower RMSE means better predictions.
    """

    return float(
        np.sqrt(np.mean((np.array(true_values) - np.array(predicted_values)) ** 2))
    )


def train_model(features):
    """
    Train the LightGBM model to predict engine remaining useful life.

    We divide the engines into training and testing groups.
    We split by engine instead of randomly splitting rows
    because readings from the same engine are very similar.

    This gives us a more realistic evaluation of the model.
    """

    # Get all the unique engine IDs
    engines = features["unit"].unique()

    # Use a fixed random seed so we get the same split every time
    rng = np.random.default_rng(42)
    shuffled_engines = rng.permutation(engines)

    # Use 25% of engines for testing and 75% for training
    test_count = int(len(engines) * 0.25)

    test_engines = shuffled_engines[:test_count]
    train_engines = shuffled_engines[test_count:]

    # Separate the data based on the selected engines
    train_data = features[features["unit"].isin(train_engines)]
    test_data = features[features["unit"].isin(test_engines)]

    # Select the columns that the model will use for prediction
    # We remove RUL because that's what we want to predict
    # We also remove the engine ID because it's just an identifier
    feature_columns = [
        column for column in features.columns
        if column not in ["rul", "unit"]
    ]

    # LightGBM is a gradient boosting model
    # It builds multiple decision trees to improve predictions
    params = {
        "objective": "regression",  # We are predicting a numerical value
        "learning_rate": 0.05,     # Controls how much each tree contributes
        "num_leaves": 31,          # Maximum number of leaves in each tree
        "verbose": -1,             # Avoid printing too many training messages
        "seed": 42,                # Makes training more reproducible
    }

    # Convert the training data into LightGBM's dataset format
    lightgbm_train_data = lgb.Dataset(
        train_data[feature_columns],
        label=train_data["rul"]
    )

    # Train the model using 300 boosting rounds
    model = lgb.train(
        params,
        lightgbm_train_data,
        num_boost_round=300
    )

    # Use the trained model to predict RUL for the test engines
    predictions = model.predict(test_data[feature_columns])

    # Make sure the predicted RUL stays between 0 and 125
    # Negative remaining life wouldn't make sense here
    predictions = np.clip(predictions, 0, RUL_CAP)

    # Calculate the prediction error
    model_rmse = rmse_score(test_data["rul"], predictions)

    # Create a simple baseline model for comparison
    # This just predicts the average RUL for every engine
    # Our LightGBM model should perform better than this
    baseline_predictions = np.full(
        len(test_data),
        train_data["rul"].mean()
    )

    baseline_rmse = rmse_score(test_data["rul"], baseline_predictions)

    # Calculate how much better LightGBM performs than the baseline
    improvement = (baseline_rmse - model_rmse) / baseline_rmse * 100

    # Print the training and testing results
    print(f"Training engines       : {len(train_engines)}")
    print(f"Testing engines        : {len(test_engines)}")
    print(f"Training rows          : {len(train_data)}")
    print(f"Testing rows           : {len(test_data)}")
    print(f"Baseline RMSE (average): {baseline_rmse:.2f} cycles")
    print(f"LightGBM RMSE          : {model_rmse:.2f} cycles")
    print(f"Improvement            : {improvement:.1f}% better than baseline")

    # Save the results so we can display them on the dashboard later
    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

    # Store actual RUL and predicted RUL for comparison
    # This can be used to create graphs in the dashboard
    evaluation = pd.DataFrame(
        {
            "actual_rul": test_data["rul"].to_numpy(),
            "predicted_rul": predictions,
        }
    )

    evaluation.to_csv(
        RESULTS_FOLDER / "evaluation.csv",
        index=False
    )

    # Find out which features were most useful for the model
    # For example, certain sensor readings might be more important
    # for predicting engine failure than others
    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importance(importance_type="gain"),
        }
    ).sort_values("importance", ascending=False)

    # Save feature importance values for the dashboard
    importance.to_csv(
        RESULTS_FOLDER / "feature_importance.csv",
        index=False
    )

    # Store the main training results in a dictionary
    # The dashboard can read this information from the JSON file
    metrics = {
        "train_engines": int(len(train_engines)),
        "test_engines": int(len(test_engines)),
        "train_rows": int(len(train_data)),
        "test_rows": int(len(test_data)),
        "baseline_rmse": round(baseline_rmse, 2),
        "model_rmse": round(model_rmse, 2),
        "improvement_percent": round(float(improvement), 1),
        "window": 20,
        "rul_cap": RUL_CAP,
        "number_of_features": len(feature_columns),
    }

    # Save the metrics in JSON format
    (RESULTS_FOLDER / "metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )

    # Display the top 10 features that contributed most to predictions
    print("\nTop 10 important features:")
    print(importance.head(10).to_string(index=False))

    return model, feature_columns, metrics


# ---------------------------------------------------------
# Main function
# This runs all the steps needed to train and save the model
# ---------------------------------------------------------

def main():

    # Step 1: Download the dataset if it's not already available
    print("Step 1/5: Checking the dataset")
    download_data()

    # Step 2: Load the engine sensor readings
    print("\nStep 2/5: Loading the data")
    data = load_data()

    # Print the total number of readings and engines
    print(f"Rows: {len(data)}, Engines: {data['unit'].nunique()}")

    # Step 3: Calculate the remaining useful life for each reading
    # This will be the target value for our machine learning model
    print("\nStep 3/5: Creating the RUL label")
    data = add_rul(data)

    # Step 4: Create features using sensor readings
    # We use information from the last 20 cycles
    # to help the model understand how engine conditions are changing
    print("\nStep 4/5: Creating features from the last 20 cycles")
    features = create_features(data)

    print(
        f"Feature rows: {len(features)}, "
        f"Features per row: {features.shape[1] - 2}"
    )

    # Step 5: Train the LightGBM model and check its performance
    print("\nStep 5/5: Training the LightGBM model")
    model, feature_columns, metrics = train_model(features)

    # Save the trained model along with the feature names
    # We also store RMSE so we know how well the model performed
    saved_data = {
        "model": model,
        "feature_columns": feature_columns,
        "rmse": metrics["model_rmse"],
    }

    # Pickle lets us save the trained model to a file
    # This way we don't need to train it again every time
    # we want to make a prediction
    with open(MODEL_FILE, "wb") as file:
        pickle.dump(saved_data, file)

    # Let the user know where the model and results were saved
    print(f"\nModel saved to {MODEL_FILE}")
    print(f"Evaluation results saved to the {RESULTS_FOLDER}/ folder")

    # The API will load our saved model and use it for predictions
    print("Next step: start the API with  uvicorn api:app --port 8000")


# Only run main() when this file is executed directly
# It won't run automatically if we import this file somewhere else
if __name__ == "__main__":
    main()