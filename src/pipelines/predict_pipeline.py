import os 
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logger
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self, features : pd.DataFrame):
        try:
            model_path = 'artifacts/model_trainer.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'

            # load model and preprocessing pipeline
            model = load_object(file_path= model_path)
            preprocessor = load_object(file_path= preprocessor_path)

            # Preprocess incoming features
            data_scaled = preprocessor.transform(features)

            # predict 
            preds = model.predict(data_scaled)

            return preds
        except Exception as e:
            raise CustomException(e, sys)
        
class CustomData:
    def __init__(self,
                  age:float,
                  gender :str,
                  total_bilirubin :float,
                  direct_bilirubin : float,
                  alkphos : float,
                  alamine: float,
                  aspartate : float,
                  total_protiens : float, 
                  albumin : float,
                  ag_ratio : float
                 ):
        self.age = age
        self.gender = gender
        self.total_bilirubin = total_bilirubin
        self.direct_bilirubin = direct_bilirubin
        self.alkphos = alkphos
        self.alamine = alamine
        self.aspartate = aspartate 
        self.total_protiens = total_protiens
        self.albumin = albumin
        self.ag_ratio = ag_ratio
    def get_data_as_data_frame(self):
        try: 
            data_dict = {
                'Age': [self.age],
                'Gender' : [self.gender],
                'Total_Bilirubin' : [self.total_bilirubin],
                'Direct_Bilirubin' : [self.direct_bilirubin],
                'Alkaline_Phosphotase' : [self.alkphos],
                'Alamine_Aminotransferase' : [self.alamine],
                'Aspartate_Aminotransferase' : [self.aspartate],
                'Total_Protiens' : [self.total_protiens],
                'Albumin' : [self.albumin],
                'Albumin_and_Globulin_Ratio' : [self.ag_ratio]
            }
            return pd.DataFrame(data_dict)
        
        except Exception as e:
            raise CustomException(e, sys)

