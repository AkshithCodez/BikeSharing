🚲 End-to-End Bike Sharing Demand Prediction 🚲
🎯 Project Overview
This project implements an end-to-end machine learning solution to predict the hourly demand for a bike-sharing program. Using a real-world dataset from Capital Bikeshare in Washington D.C., the primary objective is to build a robust regression model that accurately forecasts the total number of bike rentals (count).

The project follows a modular, production-ready code structure, encompassing the entire machine learning life cycle: from data ingestion and exploratory data analysis to advanced feature engineering, model training, and finally, deployment as an interactive web application using Streamlit. The final model achieves a high R² score, demonstrating its effectiveness in capturing complex patterns in the data.

✨ Key Features
🧱 Modular Architecture: The codebase is organized into a clean, reusable structure with separate components for data ingestion, transformation, and model training.

🛠️ Advanced Feature Engineering: Goes beyond basic features to create sophisticated inputs like:

🔄 Cyclical Time Features: Uses sine/cosine transformations for hour and month to help the model understand the cyclical nature of time.

🤝 Interaction Features: Creates features to capture the combined effect of conditions, such as the interaction between bad weather and peak commute times.

🌡️ Comfort Index: Combines temperature and humidity into a single heat_index to better represent how weather feels.

🏆 Competitive Model Evaluation: Trains and evaluates a comprehensive suite of regression models to systematically identify the best performer for this specific problem.

⚙️ Robust Preprocessing: Implements a full preprocessing pipeline using Scikit-learn, handling categorical and numerical data, scaling, and encoding.

🖥️ Interactive Web Application: Deploys the final model as a user-friendly Streamlit application where users can input conditions and receive a live demand prediction.

💻 Tech Stack
Language: Python

Libraries:

Data & Analysis: Pandas, NumPy

Machine Learning: Scikit-learn, XGBoost, LightGBM

Web Framework: Streamlit

Utilities: dill (for object serialization)

🤖 Modeling Approach
The core of this project is a robust model selection process. To ensure the highest accuracy, a wide range of regression algorithms are trained and evaluated in a competitive pipeline. The model with the highest R² score on the test set is automatically selected, tuned, and saved for deployment.

The models included in the competition are:

Linear Models:

Linear Regression

Ridge Regression

Lasso Regression

Tree-Based Ensemble Models:

Decision Tree Regressor

Random Forest Regressor

AdaBoost Regressor

Gradient Boosting Regressor

XGBoost Regressor

LightGBM Regressor (🏆 Often the top performer)

📁 Project Architecture
The project follows a modular structure to ensure scalability and maintainability.

BikeSharing/
│
├── 📂 data/
├── 📜 logs/
├── 💾 saved_models/
├── ✒️ scripts/
├── 📦 src/
│   ├── 🧩 components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   │
│   └── 🔗 pipeline/
│       ├── predict_pipeline.py
│       └── train_pipeline.py
│
├── 🚀 app.py
├── 📋 requirements.txt
└── 📄 README.md

🚀 Setup and Installation
Follow these steps to set up the project environment on your local machine.

1. Clone the Repository
git clone [https://github.com/](https://github.com/)<your-github-username>/BikeSharing.git
cd BikeSharing

2. Create a Virtual Environment
It's recommended to use a virtual environment to manage dependencies.

python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On macOS/Linux

3. Install Required Libraries
All project dependencies are listed in requirements.txt.

pip install -r requirements.txt

▶️ How to Run the Project
The project is run in two main steps: first, train the model, and then launch the web application.

Step 1: Train the Model 🧠
Execute the training pipeline script. This will perform data ingestion, transformation, and model training, and it will save the final model.pkl and preprocessor.pkl files in the saved_models/ directory.

python -m src.pipeline.train_pipeline

You will see logs printed to the console, and the script will finish by displaying the final R² score of the best model.

Step 2: Launch the Streamlit Web App 🌐
Once the model is trained, you can start the interactive web application.

streamlit run app.py

This will automatically open a new tab in your web browser at http://localhost:8501, where you can use the application to get live predictions.

📈 Model Performance
After extensive feature engineering and model tuning, the final model achieved an R² Score of 0.96 on the test set, indicating a very high level of accuracy in predicting bike rental demand.