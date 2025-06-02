import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.logger import logger
from src.exception import CustomException

@dataclass
class DataIngestionConfig:
    trian_data_path :str = os.path.join('artifacts', 'train.csv')
    test_data_path :str = os.path.join('artifacts', 'test.csv')
    raw_data_path : str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logger.info("starting data ingestion for liver disease prediction dataset")

        try:
            df = pd.read_csv('notebook/data/indian_liver_data.csv')
            logger.info("Dataset loaded successfully into DataFrame")

            ## Remove Duplicates
            initial_shape = df.shape
            df.drop_duplicates(inplace= True)
            final_shape = df.shape
            logger.info(f"Removed duplicates: {initial_shape[0]- final_shape[0]} rows dropped")

            os.makedirs(os.path.dirname(self.ingestion_config.trian_data_path),exist_ok=True)
            logger.info("Artifacts directory created")

            df.to_csv(self.ingestion_config.raw_data_path, index= False, header= True)
            logger.info("Raw data saved successfully")

            train_set, test_set = train_test_split(df, test_size= 0.2, random_state= 42, stratify=df['Dataset'])
            logger.info("Train and test split done")

            train_set.to_csv(self.ingestion_config.trian_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

            return (
                self.ingestion_config.trian_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == '__main__':
    ingestion_obj = DataIngestion()
    train_path, test_path, raw_path = ingestion_obj.initiate_data_ingestion()

    transformation_obj = DataTransformation()
    train_arr, test_arr, _ = transformation_obj.initiate_data_transformation(train_path, test_path)

    model_trainer = ModelTrainer()
    model_name, score = model_trainer.initiate_model_trainer(train_arr, test_arr)
    print(f"Best Model: {model_name}, Accuracy score: {score}")
    