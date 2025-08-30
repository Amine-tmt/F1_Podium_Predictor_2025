# Formula 1 Position Predictor

[![Python](https://img.shields.io/badge/python-v3.10.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116.1-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Deployed on Render](https://img.shields.io/badge/deployed-Render-purple.svg)](https://render.com)

A machine learning model that predicts Formula 1 driver finishing positions using historical race data from 2019-2025. Built with XGBoost and deployed as a FastAPI web service.

## 🚀 Features

- Predicts F1 race finishing positions with high accuracy
- RESTful API for real-time predictions
- GPU-optimized XGBoost model
- Comprehensive hyperparameter tuning
- Web interface for easy interaction
- Deployed on Render.com for production use

## 📊 Project Overview

This project analyzes various factors including starting grid position, circuit characteristics, driver performance, and team dynamics to predict where drivers will finish in a race.

## 🛠 Tech Stack

### Machine Learning
- **Algorithm**: XGBoost Regressor
- **Framework**: scikit-learn for preprocessing and model pipeline
- **Data Processing**: pandas for data manipulation
- **Model Serialization**: joblib for model persistence

### Web Application
- **API Framework**: FastAPI 0.116.1
- **Server**: Uvicorn 0.35.0 (ASGI server)
- **Template Engine**: Jinja2 3.1.4
- **Dependency Management**: requirements.txt

### Deployment
- **Platform**: Render.com
- **Runtime**: Python 3.10.12
- **Containerization**: Automatic deployment from GitHub

## Model Architecture

### Data Preprocessing
The model uses a comprehensive preprocessing pipeline:

```python
# Categorical features: OneHotEncoder
categorical_cols = ["Circuit Name", "Driver Name", "Team"]

# Numerical features: StandardScaler
numeric_cols = ["Year", "Starting Grid"]

# Combined preprocessing using ColumnTransformer
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_cols),
    ("num", StandardScaler(), numeric_cols)
])
```

### Model Configuration
```python
XGBRegressor(
    objective="reg:squarederror",
    n_estimators=200,
    random_state=42,
    learning_rate=0.1,
    gamma=0,
    tree_method="gpu_hist",
    predictor="gpu_predictor"
)
```

### Hyperparameter Optimization
The model underwent extensive hyperparameter tuning using GridSearchCV:

- **n_estimators**: 100, 200, 300, 400, 500
- **max_depth**: 3, 5, 7, 9, 10
- **learning_rate**: 0.01, 0.05, 0.1, 0.3
- **subsample**: 0.5, 0.7, 0.9, 1.0
- **gamma**: 0, 0.1, 0.3, 0.5
- **colsample_bytree**: 0.5, 0.7, 1.0
- **min_child_weight**: 1, 3, 5, 7

Total combinations tested: 57,600 with 3-fold cross-validation.

## Dataset

### Data Sources
- **Time Period**: 2019-2025 Formula 1 seasons
- **Total Records**: 2,486 race results
- **Features**: Year, Circuit Name, Driver Name, Team, Starting Grid Position
- **Target Variable**: Final Position

### Data Characteristics
- **Circuits**: 35 unique tracks including Monaco, Silverstone, Monza, Spa-Francorchamps
- **Drivers**: 39 different drivers from recent F1 seasons
- **Teams**: All major constructors including Mercedes, Red Bull, Ferrari, McLaren, etc.

### Data Cleaning
- Removed records with "DQ" (disqualified) and "NC" (not classified) results
- Converted Final Position to integer type
- Handled missing values through dropna()

## 📁 Project Structure

```
f1-position-predictor/
├── main.py              # FastAPI application
├── podium.pkl           # Trained model file
├── requirements.txt     # Python dependencies
├── runtime.txt          # Python version specification
├── .gitignore          # Git ignore file
└── README.md           # Project documentation
```

## 🔧 Installation

### Prerequisites
- Python 3.10.12 or compatible version
- pip package manager

### Local Setup
```bash
# Clone the repository
git clone <repository-url>
cd f1-position-predictor

# Create virtual environment
python -m venv f1-env
source f1-env/bin/activate  # On Windows: f1-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## 🚀 Deployment

This project is configured for deployment on Render.com:

1. Connect your GitHub repository to Render
2. Render will automatically detect the Python environment
3. The `runtime.txt` file specifies Python 3.10.12
4. Dependencies are installed from `requirements.txt`
5. The application starts automatically

## 📖 API Documentation

### Base URL
```
Local: http://localhost:8000
Production: https://your-app.onrender.com
```

### Endpoints

#### Health Check
```http
GET /health
```

#### Home Page
```http
GET /
```

#### Make Prediction
```http
POST /predict
Content-Type: application/json
```

**Request Body:**
```json
{
    "year": 2025,
    "circuit_name": "Monaco",
    "driver_name": "Max Verstappen",
    "team": "Red Bull Racing Honda RBPT",
    "starting_grid": 1
}
```

**Response:**
```json
{
    "predicted_position": 2.1253037
}
```

## 💻 Usage Examples

## Model Performance

### Evaluation Metrics
- **Primary Metric**: Mean Absolute Error (MAE)
- **Validation**: 3-fold cross-validation
- **Train/Test Split**: 80/20

### Performance Optimization
- GPU acceleration for training and inference
- Efficient preprocessing pipeline
- Optimized hyperparameters through extensive grid search

## Deployment Architecture

### Platform: Render.com
- Automatic deployment from GitHub repository
- Environment detection through runtime.txt
- Dependency installation via requirements.txt
- Health monitoring and auto-scaling

### Configuration Files
- `runtime.txt`: Specifies Python version (python-3.10.12)
- `requirements.txt`: Lists all dependencies
- `.gitignore`: Excludes unnecessary files from version control

## Troubleshooting

### Common Deployment Issues
1. **Pandas Compilation Error on Python 3.13+**
   - Ensure runtime.txt specifies python-3.10.12
   - Consider upgrading pandas to 2.2.3+ if using newer Python versions

2. **GPU Dependencies**
   - Model includes GPU optimization but falls back to CPU if GPU unavailable
   - No additional configuration needed for CPU-only deployment

## Usage Examples

### Basic Prediction
```python
import requests

data = {
    "year": 2025,
    "circuit_name": "Hungary",
    "driver_name": "Charles Leclerc",
    "team": "Ferrari",
    "starting_grid": 3
}

response = requests.post("http://localhost:8000/predict", json=data)
prediction = response.json()["predicted_position"]
print(f"Predicted finishing position: {prediction:.2f}")
```

### Batch Predictions
The model can process multiple predictions by calling the API endpoint multiple times or by loading the saved model directly:

```python
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("podium.pkl")

# Prepare data for prediction
prediction_data = pd.DataFrame([{
    "Year": 2025,
    "Circuit Name": "Hungary",
    "Driver Name": "Charles Leclerc",
    "Team": "Ferrari",
    "Starting Grid": 3.0
}])

# Make prediction
prediction = model.predict(prediction_data)
print(f"Predicted position: {prediction[0]:.2f}")
```

## Future Enhancements

- Integration of weather conditions and track temperature
- Real-time race data feeds
- Enhanced feature engineering with tire strategies
- Circuit-specific model variants
- Driver form and recent performance metrics
- Pit stop strategy modeling

## Contributing

Contributions are welcome. Please ensure all code follows the existing style and includes appropriate tests.

---

*Forza Ferrari! May the Prancing Horse always find its way to victory.*