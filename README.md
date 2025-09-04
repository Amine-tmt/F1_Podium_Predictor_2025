# Formula 1 Position Predictor

[![Python](https://img.shields.io/badge/python-v3.10.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116.1-green.svg)](https://fastapi.tiangolo.com/)
[![Deployed on Render](https://img.shields.io/badge/deployed-Render-purple.svg)](https://render.com)

A machine learning model that predicts Formula 1 driver finishing positions using historical race data from 2019-2025. Built with XGBoost and deployed as a FastAPI web service.

## 🚀 Features

- Predicts F1 race finishing positions with high accuracy
- RESTful API for real-time predictions
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

## Dataset

### Data Sources
- **Time Period**: 2019-2025 Formula 1 seasons
- **Total Records**: 2,503 race results
- **Features**: Year, Circuit Name, Driver Name, Team, Starting Grid Position
- **Target Variable**: Final Position

### Data Characteristics
- **Circuits**: 35 unique tracks including Monaco, Silverstone, Monza, Spa-Francorchamps
- **Drivers**: 39 different drivers from recent F1 seasons
- **Teams**: All major constructors including Mercedes, Red Bull, Ferrari, McLaren, etc.

### Data Cleaning
- Removed records with "DQ" (disqualified) and "NC" (not classified) and "DNF" (did not finish) results
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

# Install dependencies
pip install -r requirements.txt

# Run the application
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

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

## Future Enhancements

- Integration of weather conditions and track temperature
- Real-time race data feeds
- Enhanced feature engineering with tire strategies
- Circuit-specific model variants
- Driver form and recent performance metrics
- Pit stop strategy modeling

## Credits

This dataset was adapted from a public dataset shared on GitHub by [toUpperCase78](https://github.com/toUpperCase78).  
It has been cleaned, updated, and extended with additional 2025 race results.

## Contributing

Contributions are welcome. Please ensure all code follows the existing style and includes appropriate tests.

---

*🏎️Forza Ferrari! May the Prancing Horse always find its way to victory.* 