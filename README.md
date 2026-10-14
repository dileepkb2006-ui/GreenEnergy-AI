 GreenEnergy AI – Renewable Energy Prediction Web App

Python · Flask · Scikit-learn · Pandas · NumPy · SQLite · Joblib

GreenEnergy AI predicts solar energy generation, estimates wind energy production and electricity consumption, and recommends renewable energy sources using Machine Learning.
The application uses three Random Forest regression models and one Random Forest classification model, with a Flask web interface for predictions and SQLite for prediction history.

Project Structure

```text
GreenEnergy_AI/
├── app.py                      # Flask web application
├── create_datasets.py           # Generate sample datasets
├── train_models.py              # Train all ML models
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
├── .gitignore                   # Git exclusions
│
├── data/
│   ├── solar_generation.csv      # Solar training dataset
│   ├── wind_generation.csv       # Wind training dataset
│   ├── energy_consumption.csv   # Energy consumption dataset
│   └── source_recommendation.csv # Renewable source dataset
│
├── models/
│   ├── solar_generation.joblib
│   ├── wind_generation.joblib
│   ├── energy_consumption.joblib
│   ├── source_recommendation.joblib
│   └── training_metrics.json     # Model evaluation results
│
└── greenenergy.db                # SQLite prediction history
```

Dataset and model files are generated locally by running the project scripts. The database is created automatically if database initialization is implemented in `app.py`.

Installation
1. Prerequisites

- Python 3.10 or a version compatible with the project dependencies
- pip
- PyCharm or Visual Studio Code

2. Create a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

Dataset Setup

Generate the sample datasets using:

```bash
python create_datasets.py
```

The script should create the following files:

| Dataset | Purpose |
|---|---|
| `solar_generation.csv` | Solar energy prediction |
| `wind_generation.csv` | Wind energy prediction |
| `energy_consumption.csv` | Electricity demand estimation |
| `source_recommendation.csv` | Solar, wind, or hybrid classification |

The datasets are intended for educational demonstrations. Their actual features and column names must match the training script.

Model Training

Run once after generating the datasets:

```bash
python train_models.py
```

What it does:

- Loads and prepares the CSV datasets.
- Trains three Random Forest regression models.
- Trains one Random Forest classification model.
- Evaluates model performance.
- Saves the trained models in `models/`.
- Saves evaluation metrics in `models/training_metrics.json`, if implemented.

Machine Learning Models

| Model | Algorithm | Output |
|---|---|---|
| Solar Generation | Random Forest Regressor | Estimated solar output |
| Wind Generation | Random Forest Regressor | Estimated wind output |
| Energy Consumption | Random Forest Regressor | Estimated electricity demand |
| Source Recommendation | Random Forest Classifier | Solar, Wind, or Hybrid |

Actual accuracy and error metrics depend on the dataset and training implementation.

Running the App
After training the models, execute:

```bash
python app.py
```

Open your browser and visit:

http://127.0.0.1:5000

Keep the terminal running while using the application.

Features and Input Parameters

| Module | Input Features | Prediction |
|---|---|---|
| Solar Generation | Sunlight hours, temperature, panel area, cloud cover | Estimated solar energy |
| Wind Generation | Wind speed, turbine capacity, air density, turbine efficiency | Estimated wind energy |
| Energy Consumption | Occupants, floor area, temperature, appliance usage | Estimated energy demand |
| Source Recommendation | Environmental and energy-related features | Solar, Wind, or Hybrid |

The exact input fields and units depend on the implementation of `create_datasets.py`, `train_models.py`, and `app.py`.

Machine Learning Architecture

Random Forest Regressor

Library: Scikit-learn
Task: Regression
Used for solar generation, wind generation, and energy consumption prediction.

Random Forest Classifier

 Library: Scikit-learn
  Task: Classification
  Used for renewable energy source recommendation.

Workflow

```text
CSV Datasets
     ↓
Data Preprocessing
     ↓
Train-Test Split
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Save Trained Models
     ↓
Flask Web Application
     ↓
Prediction Results
     ↓
SQLite Prediction History
```

Dashboard Sections

| Section | Description |
|---|---|
| Dashboard | Main application interface |
| Solar Prediction | Solar generation estimation |
| Wind Prediction | Wind energy estimation |
| Energy Consumption | Electricity demand estimation |
| Source Recommendation | Renewable energy category prediction |
| Prediction History | Previously saved predictions, if implemented |
| Model Performance | Model evaluation metrics, if implemented |

Disclaimer

GreenEnergy AI is an educational BCA Machine Learning project. Its sample datasets and predictions are intended for demonstration purposes only. Model outputs are not guaranteed to reflect real-world energy production or consumption. Do not use the results as the sole basis for renewable energy installation, equipment selection, or financial decisions.

Credits

Built as a BCA Final-Year Machine Learning Project to demonstrate the application of Python, Flask, Machine Learning, data analysis, and database integration in renewable energy.

