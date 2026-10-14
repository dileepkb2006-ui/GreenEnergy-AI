from pathlib import Path
import sqlite3
import joblib
import pandas as pd
from flask import Flask, request, render_template_string

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
DB = BASE / "greenenergy.db"

app = Flask(__name__)

MODULES = {
    "solar": {
        "title": "Solar Generation",
        "file": "solar_generation.joblib",
        "features": ["sunlight_hours", "temperature_c", "panel_area_m2", "cloud_cover_pct"],
        "labels": ["Sunlight hours/day", "Temperature (C)", "Panel area (m2)", "Cloud cover (%)"],
        "defaults": [7, 30, 20, 20],
        "unit": "kWh/day"
    },
    "wind": {
        "title": "Wind Generation",
        "file": "wind_generation.joblib",
        "features": ["wind_speed", "turbine_capacity_kw", "air_density", "turbine_efficiency"],
        "labels": ["Wind speed (m/s)", "Turbine capacity (kW)", "Air density", "Efficiency (0-1)"],
        "defaults": [8, 100, 1.225, 0.35],
        "unit": "kWh/day"
    },
    "consumption": {
        "title": "Energy Consumption",
        "file": "energy_consumption.joblib",
        "features": ["occupants", "floor_area_m2", "temperature_c", "appliance_hours"],
        "labels": ["Occupants", "Floor area (m2)", "Temperature (C)", "Appliance hours/day"],
        "defaults": [4, 100, 28, 8],
        "unit": "kWh/day"
    },
    "recommendation": {
        "title": "Renewable Source Recommendation",
        "file": "source_recommendation.joblib",
        "features": ["sunlight_hours", "wind_speed", "daily_demand", "budget_score"],
        "labels": ["Sunlight hours/day", "Wind speed (m/s)", "Daily demand (kWh)", "Budget score (1-10)"],
        "defaults": [7, 6, 30, 7],
        "unit": ""
    }
}

PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GreenEnergy AI</title>
<style>
*{box-sizing:border-box}
body{margin:0;background:#eff7f0;color:#193d2a;font-family:Arial,sans-serif}
header{background:#116b43;color:white;padding:28px 6%}
header h1{margin:0 0 8px}
main{max-width:1150px;margin:28px auto;padding:0 18px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:18px}
.card{background:white;border-radius:14px;padding:20px;box-shadow:0 3px 15px #0000000c}
h2{font-size:20px}
label{display:block;font-size:14px;margin:12px 0 5px}
input{width:100%;padding:10px;border:1px solid #cbd8cd;border-radius:7px}
button{margin-top:16px;padding:11px 15px;background:#137c4b;color:white;border:0;border-radius:7px;cursor:pointer;font-weight:bold}
button:hover{background:#095b34}
.result{margin-top:22px;background:#e2f6e8;border:1px solid #9bd2ab}
.error{background:#fff0ee;border-color:#e1aaa2}
small{color:#617469}
</style>
</head>
<body>
<header>
<h1>GreenEnergy AI</h1>
<p>Machine Learning for a sustainable energy future</p>
</header>
<main>
<h2>Prediction Dashboard</h2>
<p>Choose a module, enter the required values, and generate a prediction.</p>
<div class="cards">
{% for key,m in modules.items() %}
<section class="card">
<h2>{{ m.title }}</h2>
<form method="post">
<input type="hidden" name="module" value="{{ key }}">
{% for i in range(m.features|length) %}
<label>{{ m.labels[i] }}</label>
<input type="number" step="any" required name="{{ m.features[i] }}" value="{{ m.defaults[i] }}">
{% endfor %}
<button type="submit">Predict</button>
</form>
</section>
{% endfor %}
</div>
{% if result is not none %}
<section class="card result">
<h2>Prediction Result</h2>
<p>{{ result_title }}</p>
<h2>{{ result }} {{ result_unit }}</h2>
<small>Educational estimate based on the project's sample training data, not a guaranteed real-world result.</small>
</section>
{% endif %}
{% if error %}
<section class="card result error"><h2>Prediction error</h2><p>{{ error }}</p></section>
{% endif %}
<section class="card" style="margin-top:22px">
<h2>Recent Predictions</h2>
{% if history %}
{% for row in history %}
<p><strong>{{ row[0] }}</strong>: {{ row[1] }} <small>({{ row[2] }})</small></p>
{% endfor %}
{% else %}
<p>No predictions yet.</p>
{% endif %}
</section>
</main>
</body>
</html>
"""

def init_db():
    with sqlite3.connect(DB) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                module TEXT NOT NULL,
                result TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

@app.route("/", methods=["GET", "POST"])
def dashboard():
    result = None
    result_title = ""
    result_unit = ""
    error = ""

    if request.method == "POST":
        key = request.form.get("module", "")
        if key not in MODULES:
            error = "Invalid prediction module."
        else:
            m = MODULES[key]
            model_path = MODELS / m["file"]
            try:
                if not model_path.is_file():
                    raise FileNotFoundError(
                        f"{m['file']} is missing. Check the models folder and run train_models.py."
                    )
                values = [float(request.form[f]) for f in m["features"]]
                if not all(pd.notna(v) for v in values):
                    raise ValueError("Please enter valid numeric values.")

                bundle = joblib.load(model_path)
                model = bundle["model"]
                model_features = bundle.get("features", m["features"])
                row = pd.DataFrame([dict(zip(m["features"], values))])
                row = row[model_features]
                prediction = model.predict(row)[0]

                result = str(prediction) if key == "recommendation" else round(float(prediction), 2)
                result_title = m["title"]
                result_unit = m["unit"]

                with sqlite3.connect(DB) as con:
                    con.execute(
                        "INSERT INTO predictions (module,result) VALUES (?,?)",
                        (m["title"], f"{result} {result_unit}")
                    )
            except Exception as exc:
                error = str(exc)

    with sqlite3.connect(DB) as con:
        history = con.execute(
            "SELECT module,result,created_at FROM predictions ORDER BY id DESC LIMIT 10"
        ).fetchall()

    return render_template_string(
        PAGE, modules=MODULES, result=result,
        result_title=result_title, result_unit=result_unit,
        error=error, history=history
    )

if __name__ == "__main__":
    init_db()
    print("GreenEnergy AI starting...")
    print("Open http://127.0.0.1:5000")
    app.run(debug=True)
