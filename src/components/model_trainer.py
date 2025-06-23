import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass 

from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,roc_curve, auc
from sklearn.model_selection import RandomizedSearchCV

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
    
    def initiate_model_trainer(self,scaled_train_array, scaled_test_array,unscaled_train_array, unscaled_test_array, preprocessor):
        try:
            logger.info("spliting into training and testing data")

            # scaled data for non-tree based models
            X_train_scaled_bal_split, y_train_scaled_bal_split = scaled_train_array[:, :-1], scaled_train_array[:, -1]
            X_test_scaled_split, y_test_scaled_split = scaled_test_array[:, :-1], scaled_test_array[:, -1]

            # unscaled data for tree based models
            X_train_unscaled_bal_split, y_train_unscaled_bal_split = unscaled_train_array[:, :-1],unscaled_train_array[:, -1]
            X_test_unscaled_split, y_test_unscaled_split = unscaled_test_array[:, :-1], unscaled_test_array[:, -1]

            ## Defining Classifiers
            non_tree_models = {
                'Logistic Regression' : LogisticRegression(max_iter=500, class_weight='balanced', random_state=42),
                'K-Nearest Neighbors' : KNeighborsClassifier()
            }
            tree_models = {
                'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
                'CatBoost' : CatBoostClassifier(class_weights=[1,3], random_state=42),
                'XGBoost' : XGBClassifier(random_state=42, scale_pos_weight=325/131),
                
            }
            all_models = {**non_tree_models, **tree_models}

            # param Grids
            param_grids = {
                'Logistic Regression' : {
                    'C' : [0.1, 1.0, 10],
                    'solver' : ['liblinear']
                },
                'K-Nearest Neighbors' : {
                    'n_neighbors' : [3,5,7],
                    'weights': ['uniform', 'distance']
                },
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
                }
            
            }

            best_model = None
            best_f1_score = 0
            best_model_name = ""
            best_false_negative = float('inf')
            for model_name, model in all_models.items():
                logger.info(f"Tuning {model_name} model")
                # Fit tree based models on unscaled data 
                if model_name in tree_models:
                    X_train, y_train = X_train_unscaled_bal_split, y_train_unscaled_bal_split
                    X_test, y_test = X_test_unscaled_split, y_test_unscaled_split

                else:
                    X_train, y_train = X_train_scaled_bal_split, y_train_scaled_bal_split
                    X_test, y_test = X_test_scaled_split, y_test_scaled_split
                
                random_search = RandomizedSearchCV( estimator=model,
                                            param_distributions=param_grids[model_name],
                                            scoring='f1_macro',  
                                            n_iter=1,
                                            cv=5,
                                            random_state=42,
                                            n_jobs=-1)
                random_search.fit(X_train, y_train)

                preds = random_search.best_estimator_.predict(X_test)
                f1 = f1_score(y_test, preds, pos_label=0, average='binary')
                report = classification_report(y_test, preds, target_names=['Liver Disease (0)', 'Healthy (1)'])
                roc_auc  = roc_auc_score(y_test, preds)
                cm = confusion_matrix(y_test, preds)
                false_negative = cm[0][1]
                logger.info(f"{model_name} F1 score (class 0): {f1} | Best Params: {random_search.best_params_}")

                # print(f"model name:{model_name}")
                # print(f"f1 score:{f1}")
                # print(f"Classification report:{report}")
                # print(f"roc auc score:{roc_auc}")
                # print(f"confusion matrix:{cm}")

                if  false_negative < best_false_negative:
                    best_false_negative = false_negative
                    best_f1_score = f1
                    best_model = random_search.best_estimator_
                    best_model_name = model_name
                    best_cm = cm
                    best_report = report

            if  best_model is None:
                raise CustomException("No suitable classification model found based on false negatives ", sys)
            
            logger.info(f"Best Model: {best_model_name} F1 Score (class 0): {best_f1_score:.4f}")
            
            
            # Save confusion matrix plot for best model
            disp = ConfusionMatrixDisplay(confusion_matrix=best_cm, display_labels=['Liver Disease (0)', 'Healthy (1)'])
            disp.plot(cmap='Blues')
            plt.title(f"Confusion Matrix: {best_model_name}")
            plt.savefig(f"artifacts/confusion_matrix_{best_model_name.replace(' ', '_')}.png")
            plt.close()

            # ROC Curve for Best Model
            fpr, tpr, _ = roc_curve(y_test, preds, pos_label=0)
            roc_auc = auc(fpr, tpr)

            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve: {best_model_name}')
            plt.legend(loc="lower right")
            plt.savefig(f'artifacts/roc_curve.png')
            plt.close()


            final_model = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', best_model)
            ])
            save_object(self.model_trainer_config.trained_model_file_path, final_model)

            return best_model_name, best_f1_score, best_cm, best_report
        
        except Exception as e:
            raise CustomException(e,sys)