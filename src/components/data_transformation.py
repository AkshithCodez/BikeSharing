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
            # Define which columns should be ordinal-encoded and which should be scaled
            numerical_columns = ["temp", "atemp", "humidity", "windspeed"]
            categorical_columns = ["season", "holiday", "workingday", "weather", "day_of_week"]

            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            # Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipelines", cat_pipeline, categorical_columns)
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "count"

            # Feature Engineering: Extract time-based features from datetime
            for df in [train_df, test_df]:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df['hour'] = df['datetime'].dt.hour
                df['day'] = df['datetime'].dt.day
                df['month'] = df['datetime'].dt.month
                df['year'] = df['datetime'].dt.year
                df['day_of_week'] = df['datetime'].dt.dayofweek # Add day of the week

            # Define features for the model
            input_feature_train_df = train_df.drop(columns=[target_column_name, 'datetime', 'casual', 'registered'], axis=1)
            # Apply the log transformation to the target variable
            target_feature_train_df = np.log1p(train_df[target_column_name])

            input_feature_test_df = test_df.drop(columns=[target_column_name, 'datetime', 'casual', 'registered'], axis=1)
            # Apply the log transformation to the target variable
            target_feature_test_df = np.log1p(test_df[target_column_name])

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)