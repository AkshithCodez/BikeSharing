
# 🚲 End-to-End Bike Sharing Demand Prediction 🎯

This project implements an **end-to-end machine learning solution** to predict the hourly demand for a bike-sharing program.
Using a real-world dataset from **Capital Bikeshare in Washington D.C.**, the primary objective is to build a robust regression model that accurately forecasts the total number of bike rentals (**count**).

The project follows a **modular, production-ready code structure**, encompassing the entire machine learning life cycle:
from **data ingestion** and **exploratory data analysis** to **advanced feature engineering**, **model training**, and finally, **deployment** as an interactive **Streamlit web application**.

The final model achieves a **high R² score**, demonstrating its effectiveness in capturing complex patterns in the data.

---

## ✨ Key Features

* 🧱 **Modular Architecture**: Clean, reusable code with separate components for data ingestion, transformation, and model training.
* 🛠️ **Advanced Feature Engineering**:

  * 🔄 *Cyclical Time Features*: Sine/cosine transformations for hour and month.
  * 🤝 *Interaction Features*: Captures combined effects (e.g., bad weather + peak commute).
  * 🌡️ *Comfort Index*: Combines temperature & humidity into a single `heat_index`.
* 🏆 **Competitive Model Evaluation**: Comprehensive suite of regression models compared to select the best performer.
* ⚙️ **Robust Preprocessing**: Full Scikit-learn pipeline for scaling, encoding, and transformation.
* 🖥️ **Interactive Web Application**: Streamlit app where users can input conditions and receive live demand predictions.

---

## 💻 Tech Stack

* **Language**: Python
* **Libraries**:

  * Data & Analysis → *Pandas, NumPy*
  * Machine Learning → *Scikit-learn, XGBoost, LightGBM*
  * Web Framework → *Streamlit*
  * Utilities → *dill (for object serialization)*

---

## 🤖 Modeling Approach

The project follows a **robust model selection process**:

1. Train multiple regression models.
2. Evaluate each using the test R² score.
3. Select the top-performing model, tune it, and save for deployment.

**Models Trained:**

* **Linear Models**: Linear Regression, Ridge, Lasso
* **Tree-Based Models**: Decision Tree, Random Forest, AdaBoost, Gradient Boosting
* **Boosting Models**: XGBoost, LightGBM *(🏆 often the best performer)*

---

## 📁 Project Architecture

```
BikeSharing/
│
├── data/                # Raw and processed data
├── logs/                # Training and evaluation logs
├── saved_models/        # Serialized models & preprocessors
├── scripts/             # Utility scripts
├── src/
│   ├── components/      # Core ML components
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   │
│   └── pipeline/        # Training & prediction pipelines
│       ├── train_pipeline.py
│       └── predict_pipeline.py
│
├── app.py               # Streamlit web application
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
```

---

## 🚀 Setup and Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/<your-github-username>/BikeSharing.git
   cd BikeSharing
   ```

2. **Create a Virtual Environment**

   ```bash
   python -m venv venv
   venv\Scripts\activate      # On Windows
   # source venv/bin/activate  # On macOS/Linux
   ```

3. **Install Required Libraries**

   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ How to Run the Project

### Step 1: Train the Model 🧠

Run the training pipeline:

```bash
python -m src.pipeline.train_pipeline
```

* Performs data ingestion, transformation, and training.
* Saves the final `model.pkl` and `preprocessor.pkl` into `saved_models/`.
* Displays the final **R² score** in the console.

### Step 2: Launch the Web App 🌐

Run the Streamlit app:

```bash
streamlit run app.py
```

* Opens in your browser at: [http://localhost:8501](http://localhost:8501)

---

## 📈 Model Performance

After **extensive feature engineering** and **hyperparameter tuning**,
the final model achieved an **R² Score of 0.96** on the test set,
indicating **very high accuracy** in predicting bike rental demand.

---

