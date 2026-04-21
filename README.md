# Predictive Maintenance Dashboard

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-green)
![FastAPI](https://img.shields.io/badge/API-FastAPI-teal)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)

A predictive maintenance project that predicts
**how many cycles a machine can still run before it fails** — its
**Remaining Useful Life (RUL)** — using NASA's C-MAPSS turbofan
engine dataset.

> **Live demo:** [Live dashboard](#deploy-the-dashboard-streamlit-cloud)
---

## The problem

Machines that fail without warning cause expensive unplanned
downtime. If we can estimate *how much useful life is left*, a
maintenance team can plan the repair before the failure happens.

This project demonstrates that idea end to end:

```
   NASA data          Training            Prediction API
┌────────────┐     ┌────────────┐      ┌──────────────────┐
│  C-MAPSS   │────▶│  train.py  │─────▶│     api.py       │
│  dataset   │     │  LightGBM  │      │  POST /predict   │
└────────────┘     └────────────┘      └────────▲─────────┘
                                                │ sensor readings
                                       ┌────────┴─────────┐
                                       │  simulator.py    │
                                       │  (live machines) │
                                       └────────┬─────────┘
                                                │ predictions
                                       ┌────────▼─────────┐
                                       │  live_data.db    │
                                       │     SQLite       │
                                       └────────┬─────────┘
                                                │
                                       ┌────────▼─────────┐
                                       │     app.py       │
                                       │    Dashboard     │
                                       └──────────────────┘
```

The simulator plays the role of real machines. In a real factory,
sensors on actual machines would send their readings to the same
API — the rest of the system would not change.

---

## Results

| Model | RMSE (cycles) |
|---|---|
| Baseline (always predict the average RUL) | 41.90 |
| **LightGBM (this project)** | **14.68** |

The LightGBM model is **65% better than the simple baseline** i.e., its prediction is wrong by about 15 cycles on
average, tested on 25 engines the model never saw during training.

---

## Project structure

```
Predictive-Maintenance/
│
├── data/                  # C-MAPSS txt files (downloaded by train.py)
├── train.py               # Train LightGBM → model.pkl + results/
├── features.py            # ONE shared feature function (training + live)
├── simulator.py           # Replay engines → ask API → save to SQLite
├── api.py                 # FastAPI: POST /predict → RUL, health, status
├── app.py                 # Streamlit: Fleet + Machine + Model pages
├── tests/
│   └── test_features.py   # Simple tests for the feature code
├── assets/                # Result charts + sample data for the demo
├── results/               # metrics.json, evaluation.csv, importance
├── Dockerfile             # One image for all services
├── docker-compose.yml     # Run everything with one command
├── requirements.txt       # Pinned package versions
└── README.md
```

Files created when you run the project (not committed to Git):

- `model.pkl` — the trained model, saved by `train.py`
- `live_data.db` — the SQLite database updated by `simulator.py`

---

## How to run

### Option 1: Run locally (recommended for learning)

**1. Install the requirements**

```bash
pip install -r requirements.txt
```

**2. Train the model** (downloads the dataset the first time)

```bash
python train.py
```

**3. Start the prediction API** — Terminal 1

```bash
uvicorn api:app --port 8000
```

Open http://localhost:8000/docs to test the API in your browser.

**4. Start the simulator** — Terminal 2

```bash
python simulator.py --machines 5 --interval 2
```

**5. Start the dashboard** — Terminal 3

```bash
streamlit run app.py
```

Open http://localhost:8501. Tick **Auto-refresh every 3 seconds**
in the sidebar while the simulator is running.

To stop the simulator, press `Ctrl+C`.

### Option 2: Run with Docker (one command)

Train once on your machine so `model.pkl` and `data/` exist, then:

```bash
docker compose up --build
```

- Dashboard: http://localhost:8501
- API docs: http://localhost:8000/docs
  
---

## Dataset credit

A. Saxena and K. Goebel, *Turbofan Engine Degradation Simulation
Data Set*, NASA Prognostics Data Repository, NASA Ames Research
Center, 2008. The dataset is downloaded by `train.py` and is not
committed to this repository.
