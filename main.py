from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained pipeline directly
model = joblib.load(os.path.join(BASE_DIR, "podium.pkl"))

# Predefined drivers and their teams
drivers_teams = {
    "Oscar Piastri": "McLaren",
    "Lando Norris": "McLaren",
    "Charles Leclerc": "Ferrari",
    "Lewis Hamilton": "Ferrari",
    "George Russell": "Mercedes",
    "Kimi Antonelli": "Mercedes",
    "Max Verstappen": "RB Racing",
    "Yuki Tsunoda": "RB Racing",
    "Alexander Albon": "Williams",
    "Carlos Sainz": "Williams",
    "Lance Stroll": "Aston Martin",
    "Fernando Alonso": "Aston Martin",
    "Nico Hülkenberg": "Kick Sauber",
    "Gabriel Bortoleto": "Kick Sauber",
    "Liam Lawson": "Racing Bulls",
    "Isack Hadjar": "Racing Bulls",
    "Esteban Ocon": "Haas",
    "Oliver Bearman": "Haas",
    "Pierre Gasly": "Alpine",
    "Franco Colapinto": "Alpine"
}

# Predefined circuits
circuits = [
    "Australia",
    "Bahrain",
    "China",
    "Azerbaijan",
    "Spain",
    "Monaco",
    "Canada",
    "France",
    "Austria",
    "Great Britain",
    "Germany",
    "Hungary",
    "Belgium",
    "Italy",
    "Singapore",
    "Russia",
    "Japan",
    "Mexico",
    "United States",
    "Brazil",
    "Abu Dhabi",
    "Styria",
    "70th Anniversary",
    "Tuscany",
    "Eifel",
    "Portugal",
    "Emilia Romagna",
    "Turkey",
    "Sakhir",
    "Netherlands",
    "Qatar",
    "Saudi Arabia",
    "Miami",
    "Las Vegas",
    "Emilia-Romagna"
]


# FastAPI setup
app = FastAPI()
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

@app.get("/", response_class=HTMLResponse)
async def form_page(request: Request):
    return templates.TemplateResponse("form.html", {
        "request": request,
        "drivers": drivers_teams,
        "circuits": circuits
    })

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request,
                  year: int = Form(...),
                  circuit: str = Form(...)):
    form_data = await request.form()

    # Build DataFrame with training column names
    features = []
    for driver, team in drivers_teams.items():
        grid_pos = int(form_data.get(driver, 0))
        features.append([year, circuit, driver, team, grid_pos])

    X = pd.DataFrame(features, columns=["Year", "Circuit Name", "Driver Name", "Team", "Starting Grid"])

    # Predict using pipeline
    preds = model.predict(X)

    # Rank predictions
    results = list(zip(drivers_teams.keys(), preds))
    results.sort(key=lambda x: x[1])  # lower predicted value = better
    podium = results[:3]

    return templates.TemplateResponse("result.html", {
        "request": request,
        "podium": podium,
        "circuit": circuit,
        "year": year
    })