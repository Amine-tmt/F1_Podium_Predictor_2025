# 🏎️ F1 Podium Predictor (2025)

A machine learning project to predict the **F1 podium results** based on drivers, teams, circuits, and grid positions.  
The app is built with **FastAPI**, uses **XGBoost** for prediction, and is styled with a modern F1-inspired UI.

---

## 🚀 Features
- Predicts **top 3 podium finishers** for a given race.
- User-friendly web interface powered by **FastAPI + Jinja2**.
- F1-themed design with a red & black podium display.
- Dynamic driver and circuit selection.

---

## 📂 Project Structure
F1_Podium_Predictor/
│── main.py # FastAPI app entry point
│── model.pkl # Trained ML model (XGBoost)
│── requirements.txt # Project dependencies
│── templates/ # HTML templates (Jinja2)
│ └── index.html
│── static/ # CSS, images, assets
│ └── style.css
│── F1_Predict.ipynb # Notebook with data prep & training
└── README.md # Project documentation


## ⚙️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR-USERNAME/F1_Podium_Predictor.git
   cd F1_Podium_Predictor