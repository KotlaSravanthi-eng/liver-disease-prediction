import os
import sys
import numpy as np
from dataclasses import dataclass 

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logger
from src.utils import save_object
import warnings 
warnings.filterwarnings('ignore')


@dataclass 
class ModelTrianerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model_trainer.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrianerConfig() 
    
    def initiate_model_trainer(self,scaled_train_array, scaled_test_array,unscaled_train_array, unscaled_test_array):
        try:
            logger.info("spliting training and test input data")

            # scaled data for non-tree based models
            X_train_scaled, y_train_scaled = scaled_train_array[:, :-1], scaled_train_array[:, -1]
            X_test_scaled, y_test_scaled = scaled_test_array[:, :-1], scaled_test_array[:, -1]

            # unscaled data for tree based models
            X_train_unscaled, y_train_unscaled = unscaled_train_array[:, :-1],unscaled_train_array[:, -1]
            X_test_unscaled, y_test_unscaled = unscaled_test_array[:, :-1], unscaled_test_array[:, -1]

            ## Defining Classifiers
            tree_models = {
                'Random Forest': RandomForestClassifier(),
                'CatBoost' : CatBoostClassifier(verbose = 0),
                'XGBoost' : XGBClassifier(eval_metric = 'logloss')
            }

            non_tree_models = {
                'Logistic Regression' : LogisticRegression(max_iter= 1000),
                'K-Nearest Neighbors' : KNeighborsClassifier()
            }

            all_models = {**tree_models, **non_tree_models}

            # param Grids
            param_grids = {
                'Random Forest': {
                    'n_estimators': [100, 200],
                    'max_depth' : [None, 10, 20]
                },
                'CatBoost' : {
                    'depth' : [6,8],
                    'learning_rate' : [0.01, 0.1]
                },
                'XGBoost' : {
                    'n_estimators' : [100, 200, 300],
                    'max_depth' : [4, 6, 7],
                    'learning_rate' : [0.05, 0.1],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree' : [0.6, 0.8]
                },
                'Logistic Regression' : {
                    'C' : [0.1, 1.0, 10],
                    'solver' : ['liblinear']
                },
                'K-Nearest Neighbors' : {
                    'n_neighbors' : [3,5,7],
                    'weights': ['uniform', 'distance']
                }
            }

            best_model = None
            best_score = 0
            best_model_name = ""
            
            for model_name, model in all_models.items():
                logger.info(f"Tuning {model_name} model")
            # Fit tree based models on unscaled data 
                if model_name in tree_models:
                    X_train, y_train = X_train_unscaled, y_train_unscaled
                    X_test, y_test = X_test_unscaled, y_test_unscaled
                else:
                    X_train, y_train = X_train_scaled, y_train_scaled
                    X_test, y_test = X_test_scaled, y_test_scaled
                
                grid = GridSearchCV(model, param_grids[model_name], cv=5, scoring='accuracy', n_jobs=-1)
                grid.fit(X_train, y_train)

                preds = grid.best_estimator_.predict(X_test)
                acc = accuracy_score(y_test, preds)

                logger.info(f"{model_name} best score: {acc} | Best Params: {grid.best_params_}")

                if acc > best_score:
                    best_score = acc
                    best_model = grid.best_estimator_
                    best_model_name = model_name

            if best_score < 0.6 or best_model is None:
                raise CustomException("No suitable classification model found with accuracy > 0.6 ")
            
            logger.info(f"Best Model: {best_model_name} with Accuracy Score: {best_score}")
            save_object(self.model_trainer_config.trained_model_file_path, best_model)

            return best_model_name, best_score
        
        except Exception as e:
            raise CustomException(e,sys)