# app.py

import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from datetime import datetime

# Streamlit page configuration
st.set_page_config(
    page_title="Bike Sharing Demand Prediction",
    page_icon="🚲",
    layout="wide"
)

st.title("🚲 Bike Sharing Demand Prediction")

# Create the input form
with st.form("prediction_form"):
    st.header("Enter the details to predict bike demand:")

    # Create columns for layout
    col1, col2, col3 = st.columns(3)

    with col1:
        # Datetime input
        d = st.date_input("Date", datetime.now())
        t = st.time_input('Time', datetime.now().time())
        dt = datetime.combine(d, t)
        
        # Season
        season = st.selectbox("Season", options=[1, 2, 3, 4], format_func=lambda x: {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}[x])
        
        # Weather
        weather = st.selectbox("Weather", options=[1, 2, 3, 4], format_func=lambda x: {1: 'Clear', 2: 'Mist/Cloudy', 3: 'Light Snow/Rain', 4: 'Heavy Rain/Snow'}[x])

    with col2:
        # Temperature
        temp = st.slider("Temperature (°C)", min_value=-10.0, max_value=45.0, value=25.0, step=0.5)
        
        # Apparent Temperature
        atemp = st.slider("Apparent Temperature (°C)", min_value=-10.0, max_value=50.0, value=28.0, step=0.5)
        
        # Humidity
        humidity = st.slider("Humidity (%)", min_value=0, max_value=100, value=60)

    with col3:
        # Windspeed
        windspeed = st.slider("Windspeed (km/h)", min_value=0.0, max_value=60.0, value=15.0, step=0.5)
        
        # Holiday
        holiday = st.radio("Is it a holiday?", options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
        
        # Working Day
        workingday = st.radio("Is it a working day?", options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    
    # Submit button
    submit_button = st.form_submit_button(label='Predict Demand')

# Prediction logic
if submit_button:
    data = CustomData(
        season=season,
        holiday=holiday,
        workingday=workingday,
        weather=weather,
        temp=temp,
        atemp=atemp,
        humidity=humidity,
        windspeed=windspeed,
        hour=dt.hour,
        day=dt.day,
        month=dt.month,
        year=dt.year
    )
    
    pred_df = data.get_data_as_data_frame()
    
    st.write("Input Data:")
    st.dataframe(pred_df)
    
    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(pred_df)
    
    st.success(f"The predicted bike rental count is: {int(results[0])}")