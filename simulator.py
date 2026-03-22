import argparse
import os
import random
import sqlite3
import time
import pandas as pd
import requests

from pathlib import Path
from features import ALL_COLUMNS, SENSORS, WINDOW

# dataset path
DATA_FILE = Path("data/train_FD001.txt")

# SQLite database where simulated readings will be stored
DATABASE_FILE = Path("live_data.db")

# The dataset contains 21 different sensor readings
ALL_SENSORS = [f"s{i}" for i in range(1, 22)]

# Address of our prediction API
# By default, it runs locally on port 8000
# Docker can provide a different address using API_URL
DEFAULT_API_URL = os.environ.get(
    "API_URL",
    "http://127.0.0.1:8000/predict"
)

def load_dataset():
    """Load the NASA engine dataset that we use for simulation."""

    # Check if the dataset exists before trying to read it
    # The train.py file downloads this dataset automatically
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Dataset not found. Please run train.py first. "
            "It downloads the data."
        )

    # Read the dataset into a pandas DataFrame
    # The values are separated by spaces
    data = pd.read_csv(
        DATA_FILE,
        sep=r"\s+",
        header=None,
        names=ALL_COLUMNS,
    )

    # Sort the readings by engine number and cycle
    # This makes sure we simulate the engine readings in the correct order
    return data.sort_values(["unit", "cycle"]).reset_index(drop=True)

def predict_rul(history, api_url):
    """
    Send the recent sensor readings of a machine to the API.

    The API uses our trained LightGBM model to predict:
    - Remaining Useful Life (RUL)
    - Machine health
    - Machine status
    """

    # Select the cycle number and sensor readings needed by the API
    columns_to_send = ["cycle"] + SENSORS

    # Convert the sensor data into a dictionary
    # This is needed because we send the data as JSON
    payload = {
        "history": history[columns_to_send].to_dict(orient="records")
    }

    try:
        # Send the sensor readings to the prediction API
        # The API processes the data and returns a prediction
        response = requests.post(
            api_url,
            json=payload,
            timeout=10
        )

    except requests.ConnectionError:
        # Show a useful error if the API server is not running
        raise ConnectionError(
            f"Could not connect to the API at {api_url}. "
            "Please start it first with:  uvicorn api:app --port 8000"
        )

    # Check if the API returned an error
    response.raise_for_status()

    # Convert the API response from JSON into a Python dictionary
    return response.json()

def create_database():
    """
    Create a SQLite database to store the simulated machine readings.

    Each row will contain the machine name, cycle number,
    predicted RUL, health, status, and sensor values.
    """

    # Create database columns for all 21 sensors
    # We use REAL because sensor readings contain decimal values
    sensor_columns = ", ".join(
        f"{sensor} REAL" for sensor in ALL_SENSORS
    )

    # Connect to the SQLite database
    # SQLite creates the file automatically if it doesn't exist
    connection = sqlite3.connect(DATABASE_FILE)

    # Remove the old readings table if it already exists
    # This allows every simulation run to start with fresh data
    connection.execute("DROP TABLE IF EXISTS readings")

    # Create a new table for storing machine readings
    connection.execute(
        f"""
        CREATE TABLE readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            machine_id TEXT,
            cycle INTEGER,
            rul REAL,
            health REAL,
            status TEXT,
            {sensor_columns}
        )
        """
    )

    # Save the database changes
    connection.commit()

    return connection


def save_reading(connection, machine_id, current_row, prediction):
    """
    Save the current machine reading and its prediction
    into the SQLite database.
    """

    # Get all 21 sensor values from the current engine cycle
    # Convert them to float so they can be stored in SQLite
    sensor_values = [
        float(current_row[sensor]) for sensor in ALL_SENSORS
    ]

    # Prepare the sensor column names for the SQL query
    columns = ", ".join(ALL_SENSORS)

    # Create placeholders for inserting sensor values
    # Using placeholders is safer than putting values directly into SQL
    placeholders = ", ".join("?" for _ in ALL_SENSORS)

    # Insert the machine details and sensor readings into the database
    connection.execute(
        f"""
        INSERT INTO readings
            (machine_id, cycle, rul, health, status, {columns})
        VALUES
            (?, ?, ?, ?, ?, {placeholders})
        """,
        [
            machine_id,
            int(current_row["cycle"]),
            prediction["rul"],
            prediction["health"],
            prediction["status"],
        ]
        + sensor_values,
    )

    # We don't commit here because the main loop saves
    # all machine readings together after each simulation step

def main():

    # argparse lets us change simulation settings from the terminal
    # For example, we can choose how many machines to simulate
    # and how quickly they send readings
    parser = argparse.ArgumentParser(
        description="Simulate live machines"
    )

    # Number of machines to simulate
    # By default, we simulate 5 machines
    parser.add_argument(
        "--machines",
        type=int,
        default=5,
        help="number of machines to simulate (default: 5)"
    )

    # Time gap between simulation steps
    # Default is 2 seconds
    parser.add_argument(
        "--interval",
        type=float,
        default=2,
        help="seconds to wait between cycles (default: 2)"
    )

    # Number of simulation steps
    # Setting this to 0 means the simulator keeps running
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="stop after this many steps (default: 0 = run forever)"
    )

    # API address where sensor data will be sent
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help="prediction API address"
    )

    # Read the settings provided by the user
    args = parser.parse_args()

    # Load the engine dataset
    print("Loading the dataset...")
    data = load_dataset()

    # Create a fresh database to store the simulation results
    connection = create_database()

    print(f"Created a fresh database: {DATABASE_FILE}")
    print(f"Prediction API: {args.api_url}")

    # Use a fixed random seed so machine starting points
    # remain the same whenever we restart the simulator
    rng = random.Random(42)

    # This list will store information about each machine
    machines = []

    # Select the engines we want to use from the dataset
    # For example, if machines=3, we use the first 3 engines
    engine_numbers = sorted(data["unit"].unique())[: args.machines]

    for number, engine_number in enumerate(engine_numbers, start=1):

        # Get all the readings belonging to this engine
        engine_data = data[
            data["unit"] == engine_number
        ].reset_index(drop=True)

        # Start each machine at a random point in its lifetime
        # We only choose from the first 60% of its data
        # This helps us show machines with different health conditions
        start_position = rng.randint(
            0,
            int(len(engine_data) * 0.6)
        )

        # Store the machine name, data, and current position
        machines.append(
            {
                "name": f"Machine-{number}",
                "data": engine_data,
                "position": start_position,
            }
        )

    print(f"Starting {len(machines)} machines...")
    print("Open the dashboard with:  streamlit run app.py")
    print("Press Ctrl+C to stop.\n")

    # Keep track of how many simulation steps have completed
    step = 0

    try:
        # Keep running until the user stops the program
        # or the selected number of steps is reached
        while True:

            step += 1

            # Go through each machine and simulate its next cycle
            for machine in machines:

                engine_data = machine["data"]
                position = machine["position"]

                # Check if the machine has reached the end of its lifetime
                # In the dataset, the last cycle represents engine failure
                if position >= len(engine_data):

                    # Restart the machine from its first cycle
                    # For this project, we treat this as a repaired machine
                    # starting a new operating life
                    machine["position"] = 0
                    position = 0

                    print(
                        f"{machine['name']} restarted after reaching end of life."
                    )

                # Find the starting point of the recent sensor history
                # We use up to the last 20 cycles for prediction
                history_start = max(
                    0,
                    position - WINDOW + 1
                )

                # Get the most recent sensor readings
                # These will be sent to the prediction API
                history = engine_data.iloc[
                    history_start : position + 1
                ]

                # Get the current cycle's sensor readings
                current_row = engine_data.iloc[position]

                # Send the recent readings to the API
                # The API returns predicted RUL, health, and status
                prediction = predict_rul(
                    history,
                    args.api_url
                )

                # Save the current reading and prediction
                # so the dashboard can display them
                save_reading(
                    connection,
                    machine["name"],
                    current_row,
                    prediction
                )

                # Move the machine to its next cycle
                # This will be used in the next simulation step
                machine["position"] = position + 1

            # Save all machine readings for this step into SQLite
            connection.commit()

            print(
                f"Step {step}: saved predictions for {len(machines)} machines"
            )

            # Stop the simulator if we reached the requested number of steps
            # If steps=0, the simulator will continue running
            if args.steps and step >= args.steps:
                break

            # Wait before generating the next set of readings
            # Setting interval=0 makes the simulation run without delays
            if args.interval > 0:
                time.sleep(args.interval)

    except KeyboardInterrupt:
        # Allows us to stop the simulator using Ctrl+C
        # without displaying a long error message
        print("\nSimulator stopped.")

    # Close the database connection when the simulation finishes
    connection.close()

    print(f"Readings are saved in {DATABASE_FILE}")


# Run the simulator only when this file is executed directly
# This prevents it from starting automatically if imported elsewhere
if __name__ == "__main__":
    main()