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
            
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            data_scaled = preprocessor.transform(features)
            preds_log = model.predict(data_scaled)
            
            preds = np.expm1(preds_log)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 season: int, holiday: int, workingday: int, weather: int,
                 temp: float, atemp: float, humidity: int, windspeed: float,
                 year: int, month: int, day: int, hour: int):
        
        self.season = season
        self.holiday = holiday
        self.workingday = workingday
        self.weather = weather
        self.temp = temp
        self.atemp = atemp
        self.humidity = humidity
        self.windspeed = windspeed
        self.year = year
        self.month = month
        self.day = day
        self.hour = hour

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "season": [self.season], "holiday": [self.holiday], "workingday": [self.workingday],
                "weather": [self.weather], "temp": [self.temp], "atemp": [self.atemp],
                "humidity": [self.humidity], "windspeed": [self.windspeed], "year": [self.year]
            }
            
            df = pd.DataFrame(custom_data_input_dict)

            # --- ADD THE SAME FEATURE ENGINEERING AS IN TRAINING ---
            date_str = f"{self.year}-{self.month}-{self.day}"
            dt_obj = pd.to_datetime(date_str)
            df['day_of_week'] = dt_obj.dayofweek
            
            df['is_rush_hour'] = 1 if (7 <= self.hour <= 9) or (16 <= self.hour <= 18) else 0
            df['is_bad_weather'] = 1 if self.weather in [3, 4] else 0
            df['peak_bad_weather_interaction'] = df['is_rush_hour'] * df['is_bad_weather']

            # Cyclical Features
            df['hour_sin'] = np.sin(2 * np.pi * self.hour / 24.0)
            df['hour_cos'] = np.cos(2 * np.pi * self.hour / 24.0)
            df['month_sin'] = np.sin(2 * np.pi * self.month / 12.0)
            df['month_cos'] = np.cos(2 * np.pi * self.month / 12.0)
            
            # Comfort Index
            df['heat_index'] = self.temp * self.humidity
            
            return df

        except Exception as e:
            raise CustomException(e, sys)