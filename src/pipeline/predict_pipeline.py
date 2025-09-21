# src/pipeline/predict_pipeline.py

import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'saved_models/model.pkl'
            preprocessor_path = 'saved_models/preprocessor.pkl'
            
            # Load the saved model and preprocessor objects
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            # Scale the features and make a prediction
            data_scaled = preprocessor.transform(features)
            preds_log = model.predict(data_scaled)
            
            # Reverse the log transformation to get the actual count
            preds = np.expm1(preds_log)
            
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 season: int,
                 holiday: int,
                 workingday: int,
                 weather: int,
                 temp: float,
                 atemp: float,
                 humidity: int,
                 windspeed: float,
                 hour: int,
                 day: int,
                 month: int,
                 year: int):
        
        self.season = season
        self.holiday = holiday
        self.workingday = workingday
        self.weather = weather
        self.temp = temp
        self.atemp = atemp
        self.humidity = humidity
        self.windspeed = windspeed
        self.hour = hour
        self.day = day
        self.month = month
        self.year = year

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "season": [self.season],
                "holiday": [self.holiday],
                "workingday": [self.workingday],
                "weather": [self.weather],
                "temp": [self.temp],
                "atemp": [self.atemp],
                "humidity": [self.humidity],
                "windspeed": [self.windspeed],
                "hour": [self.hour],
                "day": [self.day],
                "month": [self.month],
                "year": [self.year],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)