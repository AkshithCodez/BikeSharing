# src/components/data_transformation.py

import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('saved_models', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_columns = [
                "temp", "atemp", "humidity", "windspeed", "is_rush_hour", 
                "is_bad_weather", "peak_bad_weather_interaction",
                "hour_sin", "hour_cos", "month_sin", "month_cos", "heat_index"
            ]
            categorical_columns = [
                "season", "holiday", "workingday", "weather", "day_of_week", "year"
            ]

            num_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
            cat_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("one_hot_encoder", OneHotEncoder()), ("scaler", StandardScaler(with_mean=False))])

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipelines", cat_pipeline, categorical_columns)
            ])
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            preprocessing_obj = self.get_data_transformer_object()
            target_column_name = "count"

            for df in [train_df, test_df]:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df['hour'] = df['datetime'].dt.hour
                df['month'] = df['datetime'].dt.month
                df['day'] = df['datetime'].dt.day
                df['year'] = df['datetime'].dt.year
                df['day_of_week'] = df['datetime'].dt.dayofweek
                
                df['is_rush_hour'] = df['hour'].apply(lambda x: 1 if (7 <= x <= 9) or (16 <= x <= 18) else 0)
                df['is_bad_weather'] = df['weather'].apply(lambda x: 1 if x in [3, 4] else 0)
                df['peak_bad_weather_interaction'] = df['is_rush_hour'] * df['is_bad_weather']

                df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
                df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
                df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
                df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
                
                df['heat_index'] = df['temp'] * df['humidity']

            # 'year' is NOT dropped, as it's used as a categorical feature
            input_feature_train_df = train_df.drop(columns=[target_column_name, 'datetime', 'casual', 'registered', 'hour', 'month', 'day'], axis=1)
            target_feature_train_df = np.log1p(train_df[target_column_name])

            input_feature_test_df = test_df.drop(columns=[target_column_name, 'datetime', 'casual', 'registered', 'hour', 'month', 'day'], axis=1)
            target_feature_test_df = np.log1p(test_df[target_column_name])

            logging.info("Applying preprocessing object.")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessing_obj)
            logging.info("Saved preprocessing object.")

            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)
        except Exception as e:
            raise CustomException(e, sys)