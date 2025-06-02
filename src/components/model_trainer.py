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

from src.exception import CustomException
from src.logger import logger
from utils import save_object

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
                'XGBoost' : XGBClassifier(use_label_encoder = False, eval_metric = 'logloss')
            }

            non_tree_models = {
                'Logistic Regression' : LogisticRegression(max_iter= 1000),
                'K-Nearest Neighbors' : KNeighborsClassifier()
            }

            all_models = {**tree_models, **non_tree_models}
            model_scores = {}

            # Fit tree based models on unscaled data 
            for model_name, model in tree_models.items():
                model.fit(X_train_unscaled, y_train_unscaled)
                preds = model.predict(X_test_unscaled)
                acc = accuracy_score(y_test_unscaled, preds)
                model_scores[model_name] = acc

            # Fit non-tree-based models on scaled data 
            for model_name, model in non_tree_models.items():
                model.fit(X_train_scaled,y_train_scaled)
                preds = model.predict(X_test_scaled)
                acc = accuracy_score(y_test_scaled, preds)
                model_scores[model_name] = acc
            # pick the best model
            best_model_name = max(model_scores, key = model_scores.get)
            best_model = all_models[best_model_name]
            best_score = model_scores[best_model_name]

            if best_score < 0.6:
                raise CustomException("No suitable classification model found with accuracy > 0.6 ")
            logger.info(f"Best Model: {best_model_name} with Accuracy Score: {best_score}")
            save_object(self.model_trainer_config.trained_model_file_path, best_model)

            return best_model_name, best_score
        
        except Exception as e:
            raise CustomException(e,sys)