import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

from src.logger import logger
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTrasformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTrasformationConfig()
    
    def get_data_transformation_object(self, numerical_columns, categorical_columns):
        try:
            num_pipeline = Pipeline(
                steps= [
                    ("imputer",SimpleImputer(strategy="median")),
                    ("power_transform", PowerTransformer(method='yeo-johnson')),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps= [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num", num_pipeline, numerical_columns),
                    ("cat", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logger.info("Loaded train and test data")

            target_column = "Dataset"
            numerical_columns = ['Age', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
                                 'Alamine_Aminotransferase','Aspartate_Aminotransferase',
                                 'Total_Proteins', 'Albumin', 'Albumin_and_Globulin_Ratio']
            
            categorical_columns = ['Gender']

            preprocessor = self.get_data_transformer_object(numerical_columns, categorical_columns)

            # separate input and target 
            X_train = train_df.drop(columns = [target_column])
            y_train = train_df[target_column]
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            logger.info("Applying preprocessing to training data")
            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled = preprocessor.transform(X_test)

            logger.info("Applying SMOTE for class imbalance")
            smote = SMOTE(random_state = 42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled,y_train)
            
            logger.info("preparing unscaled data for tree models")
            num_imputer = SimpleImputer(strategy= 'median')
            cat_imputer = SimpleImputer(strategy= 'most_frequent')
            X_train_unscaled = X_train.copy()
            X_test_unscaled = X_test.copy()

            X_train_unscaled[numerical_columns] = num_imputer.fit_transform(X_train_unscaled[numerical_columns])
            X_train_unscaled[categorical_columns] = cat_imputer.fit_transform(X_train_unscaled[categorical_columns])
            X_test_unscaled[numerical_columns] = num_imputer.fit_transform(X_test_unscaled[numerical_columns])
            X_test_unscaled[categorical_columns] = cat_imputer.fit_transform(X_test_unscaled[categorical_columns])

            # converting categorical to one hot encoding
            encoder = OneHotEncoder(handle_unknown='ignore', sparse= False)
            X_train_unscaled_cat = encoder.fit_transform(X_train_unscaled[categorical_columns])
            X_test_unscaled_cat = encoder.fit_transform(X_test_unscaled[categorical_columns])

            X_train_unscaled_final = np.hstack((X_train_unscaled[numerical_columns].values, X_test_unscaled_cat))
            X_test_unscaled_final = np.hstack((X_test_unscaled[numerical_columns].values, X_test_unscaled_cat))
            
            # save the preprocessor object
            save_object(self.data_transformation_config.preprocessor_obj_file_path, preprocessor)
            logger.info("Preprocessor saved")

            return (
                np.c_[X_train_balanced, y_train_balanced],
                np.c_[X_test_scaled, y_test],
                np.c_[X_train_unscaled_final, y_train_balanced],
                np.c_[X_test_unscaled_final, y_test.values],
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)