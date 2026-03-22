import json
import sqlite3
import time
import pandas as pd
import plotly.express as px
import streamlit as st

from pathlib import Path

# SQLite database containing the simulated machine readings
DATABASE_FILE = Path("live_data.db")

# Sample data used when the live database is not available
SAMPLE_FILE = Path("assets/sample_data.csv")

# Folder containing our model evaluation results
RESULTS_FOLDER = Path("results")

# List of all 21 sensors in the NASA engine dataset
ALL_SENSORS = [f"s{i}" for i in range(1, 22)]

# Set the dashboard title and make it use the full screen width
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    layout="wide",
)

def load_all_readings():
    """
    Load machine readings from our SQLite database.

    If the database is not available, we use the sample CSV file.
    This is useful when showing the dashboard without running
    the simulator.
    """

    # First check if the live database exists
    if DATABASE_FILE.exists():
        try:
            # Connect to the SQLite database
            connection = sqlite3.connect(DATABASE_FILE)

            # Read all machine readings into a pandas DataFrame
            data = pd.read_sql(
                "SELECT * FROM readings",
                connection
            )

            # Close the database connection after reading
            connection.close()

            # If we have readings, use the live data
            if not data.empty:
                return data, "live"

        except Exception:
            # If there is a problem reading the database,
            # we try loading the sample data instead
            pass

    # Use sample data if the simulator hasn't created live data yet
    if SAMPLE_FILE.exists():
        return pd.read_csv(SAMPLE_FILE), "sample"

    # Return an empty DataFrame if no data is available
    return pd.DataFrame(), "none"


def get_latest(data):
    """
    Get the latest reading for each machine.

    We sort machines by predicted RUL so that machines
    closer to failure appear at the top.
    """

    # Sort readings by ID and take the last reading of each machine
    latest = (
        data.sort_values("id")
        .groupby("machine_id")
        .tail(1)
    )

    # Show machines with the lowest remaining life first
    return latest.sort_values("rul").reset_index(drop=True)

def show_fleet_page(data):
    """
    Display the overall health of all simulated machines.

    This page helps us quickly identify which machines
    are healthy and which ones may need maintenance.
    """

    st.title("Fleet Overview")

    st.write(
        "Every machine's latest predicted Remaining Useful Life (RUL). "
        "Machines with the lowest RUL are shown first because they need "
        "attention first."
    )

    # Get the most recent prediction for every machine
    latest = get_latest(data)

    # Count machines based on their predicted status
    healthy = len(latest[latest["status"] == "Healthy"])
    warning = len(latest[latest["status"] == "Warning"])
    critical = len(latest[latest["status"] == "Critical"])

    # Create four columns to display the main numbers
    column1, column2, column3, column4 = st.columns(4)

    # Display total machines and their health categories
    column1.metric("Total Machines", len(latest))
    column2.metric("Healthy", healthy)
    column3.metric("Warning", warning)
    column4.metric("Critical", critical)

    st.subheader("Latest machine status")

    # Explain how machine status is decided using RUL
    st.write(
        "Healthy = RUL above 50 cycles | "
        "Warning = RUL 20 to 50 cycles | "
        "Critical = RUL below 20 cycles"
    )

    # Show the latest readings in a table
    # We only display the columns that are useful for monitoring
    st.dataframe(
        latest[["machine_id", "cycle", "rul", "health", "status"]],
        hide_index=True,
        use_container_width=True,
        column_config={
            "machine_id": "Machine",
            "cycle": "Current Cycle",
            "rul": st.column_config.NumberColumn(
                "Predicted RUL (cycles)",
                format="%.1f"
            ),
            "health": st.column_config.NumberColumn(
                "Health (%)",
                format="%.1f"
            ),
            "status": "Status",
        },
    )

    st.subheader("Predicted RUL by machine")

    # Create a bar chart to compare remaining life across machines
    # The bar colors represent the machine health status
    chart = px.bar(
        latest,
        x="machine_id",
        y="rul",
        color="status",
        title="Lower RUL means the machine is closer to failure",
        labels={
            "machine_id": "Machine",
            "rul": "Predicted RUL (cycles)"
        },
        color_discrete_map={
            "Healthy": "#22c55e",   # Green for healthy machines
            "Warning": "#f59e0b",   # Orange for warning
            "Critical": "#ef4444",  # Red for critical machines
        },
    )

    # Display the chart on the dashboard
    st.plotly_chart(chart, use_container_width=True)

def show_machine_page(data):
    """
    Show detailed information about one selected machine.

    We can check its current health, predicted remaining life,
    and how its sensor readings are changing over time.
    """

    st.title("Machine Detail")

    # Get the latest readings of all machines
    latest = get_latest(data)

    # Let the user select which machine they want to inspect
    machine_id = st.selectbox(
        "Choose a machine",
        latest["machine_id"].tolist()
    )

    # Find the latest information for the selected machine
    machine = latest[
        latest["machine_id"] == machine_id
    ].iloc[0]

    # Display the machine's current status and predictions
    column1, column2, column3, column4 = st.columns(4)

    column1.metric("Status", machine["status"])
    column2.metric("Predicted RUL", f"{machine['rul']:.1f} cycles")
    column3.metric("Health", f"{machine['health']:.1f}%")
    column4.metric("Current Cycle", int(machine["cycle"]))

    # Show a message depending on the machine's health status
    # This helps the user quickly understand if maintenance is needed
    if machine["status"] == "Critical":
        st.error(
            "This machine is critical. Maintenance should be planned soon."
        )

    elif machine["status"] == "Warning":
        st.warning(
            "This machine is in warning state. Please watch it closely."
        )

    else:
        st.success("This machine looks healthy.")

    # Get the previous readings of the selected machine
    # We only keep the latest 300 readings to avoid very large charts
    history = (
        data[data["machine_id"] == machine_id]
        .sort_values("id")
        .tail(300)
        .reset_index(drop=True)
    )

    # Stop here if there are no readings for the machine
    if history.empty:
        return

    # Give each reading a number starting from 1
    # This is useful because machines can restart after failure
    # and their cycle numbers may start again from the beginning
    history["reading"] = range(1, len(history) + 1)

    st.subheader("How the predicted RUL is changing")

    # Create a line chart showing how the predicted remaining life
    # changes as the machine goes through more cycles
    rul_chart = px.line(
        history,
        x="reading",
        y="rul",
        title=f"Predicted RUL for {machine_id}",
        labels={
            "reading": "Reading Number",
            "rul": "Predicted RUL (cycles)"
        },
    )

    st.plotly_chart(rul_chart, use_container_width=True)

    st.subheader("Sensor trends")

    # Let the user choose which sensors they want to see
    # We select four sensors by default
    selected_sensors = st.multiselect(
        "Choose sensors to show",
        ALL_SENSORS,
        default=["s2", "s4", "s11", "s14"],
    )

    # Only create the chart if at least one sensor is selected
    if selected_sensors:

        # Plot the selected sensor readings over time
        # This helps us see how sensor values change as the engine ages
        sensor_chart = px.line(
            history,
            x="reading",
            y=selected_sensors,
            title=f"Sensor readings for {machine_id}",
            labels={
                "reading": "Reading Number",
                "value": "Sensor Value"
            },
        )

        st.plotly_chart(sensor_chart, use_container_width=True)

    st.subheader("Recent readings")

    # Show the main prediction values and up to four selected sensors
    columns_to_show = [
        "reading",
        "cycle",
        "rul",
        "health",
        "status"
    ] + selected_sensors[:4]

    # Display only the most recent 20 readings
    st.dataframe(
        history[columns_to_show].tail(20),
        hide_index=True,
        use_container_width=True,
    )

def show_model_page():
    """
    Display how well our LightGBM model performs.

    The results are generated during training and saved
    in the results folder by train.py.
    """

    st.title("Model Performance")

    st.write(
        "This page shows how accurate the model is on engines it never "
        "saw during training. The results come from the files that "
        "train.py saves in the results/ folder."
    )

    # Location of the JSON file containing training results
    metrics_file = RESULTS_FOLDER / "metrics.json"

    # Check if the model has been trained
    if not metrics_file.exists():
        st.warning("No results yet. Please run train.py first.")
        return

    # Load the model evaluation results from JSON
    metrics = json.loads(metrics_file.read_text())

    # Create four columns to display the main evaluation metrics
    column1, column2, column3, column4 = st.columns(4)

    # RMSE tells us how far the model's predictions are
    # from the actual remaining useful life
    # A smaller RMSE means better predictions
    column1.metric(
        "LightGBM RMSE",
        f"{metrics['model_rmse']:.2f} cycles"
    )

    # Baseline RMSE comes from a simple model that always
    # predicts the average remaining useful life
    column2.metric(
        "Baseline RMSE",
        f"{metrics['baseline_rmse']:.2f} cycles",
        help="A 'dumb' model that always predicts the average RUL.",
    )

    # Show how much better LightGBM performs compared to the baseline
    column3.metric(
        "Improvement",
        f"{metrics['improvement_percent']:.1f}%"
    )

    # Show how many engines were used for testing
    # These engines were not included in the training data
    column4.metric(
        "Test Engines",
        metrics["test_engines"],
        help="Engines kept aside and never used in training.",
    )

    # Explain the evaluation results in a simple way
    st.info(
        f"RMSE means: on average, the prediction is wrong by about "
        f"{metrics['model_rmse']:.0f} cycles. The data was split by "
        f"engine ({metrics['train_engines']} engines for training, "
        f"{metrics['test_engines']} for testing) so similar cycles "
        f"from the same engine cannot appear in both sets."
    )

    # Load the file containing actual and predicted RUL values
    evaluation_file = RESULTS_FOLDER / "evaluation.csv"

    if evaluation_file.exists():

        evaluation = pd.read_csv(evaluation_file)

        # If there are too many readings, randomly select 1000
        # This makes the graph faster to load
        if len(evaluation) > 1000:
            evaluation = evaluation.sample(
                1000,
                random_state=42
            )

        st.subheader("Actual RUL vs Predicted RUL")

        st.write(
            "A perfect model would put every dot on the diagonal line."
        )

        # Create a scatter plot comparing actual and predicted RUL
        # Points closer to the diagonal line mean better predictions
        chart = px.scatter(
            evaluation,
            x="actual_rul",
            y="predicted_rul",
            opacity=0.5,
            labels={
                "actual_rul": "Actual RUL (cycles)",
                "predicted_rul": "Predicted RUL (cycles)",
            },
        )

        # Add a diagonal reference line
        # This line represents perfect predictions
        chart.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=125,
            y1=125,
            line={
                "color": "red",
                "dash": "dash"
            },
        )

        st.plotly_chart(chart, use_container_width=True)

    # This file tells us which features were most useful
    # when the LightGBM model made predictions
    importance_file = RESULTS_FOLDER / "feature_importance.csv"

    if importance_file.exists():

        # Read the feature importance values
        # We only display the 10 most important features
        importance = pd.read_csv(
            importance_file
        ).head(10)

        st.subheader("Top 10 important features")

        st.write(
            "These are the features the model uses most. Most of them "
            "are the average (_mean) sensor values from the last "
            "20 cycles — the general level of a sensor matters more "
            "than a single reading."
        )

        # Create a horizontal bar chart
        # Longer bars mean the feature had more importance
        chart = px.bar(
            importance.iloc[::-1],
            x="importance",
            y="feature",
            orientation="h",
            labels={
                "importance": "Importance",
                "feature": "Feature"
            },
        )

        st.plotly_chart(chart, use_container_width=True)

# Create the sidebar title
st.sidebar.title("Predictive Maintenance")

# Let the user choose which dashboard page to open
page = st.sidebar.radio(
    "Choose a page",
    [
        "Fleet Overview",
        "Machine Detail",
        "Model Performance"
    ],
)

# Add a line to separate the menu from the status information
st.sidebar.divider()

# Show the RUL limits used for deciding machine health
st.sidebar.write("**Simple status rules**")
st.sidebar.write("Healthy: RUL > 50 cycles")
st.sidebar.write("Warning: RUL 20-50 cycles")
st.sidebar.write("Critical: RUL < 20 cycles")

# Load the machine readings
# The function uses live data if available, otherwise sample data
data, source = load_all_readings()

# Show a message when we are displaying sample data
# This helps users know that the simulator is not providing live readings
if source == "sample":
    st.sidebar.info(
        "Showing sample data. Run the API and simulator "
        "for live data."
    )

# Give the user an option to automatically refresh the dashboard
# When enabled, the page updates every 3 seconds
auto_refresh = st.sidebar.checkbox(
    "Auto-refresh every 3 seconds"
)

# Show the model evaluation page
if page == "Model Performance":
    show_model_page()

# If no machine data is available, show a warning
elif data.empty:
    st.title(page)

    st.warning(
        "No data yet. Please run train.py, start the API, and run "
        "simulator.py, then refresh this page."
    )

# Show the fleet overview page
elif page == "Fleet Overview":
    show_fleet_page(data)

# Otherwise, show the selected machine's details
else:
    show_machine_page(data)

# If auto-refresh is enabled, wait 3 seconds
# and run the dashboard again to get updated readings
if auto_refresh:
    time.sleep(3)
    st.rerun()