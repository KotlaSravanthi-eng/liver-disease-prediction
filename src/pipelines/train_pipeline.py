from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


if __name__ == '__main__':
    ## Step 1:Ingest Data
    ingestion_obj = DataIngestion()
    train_path, test_path, raw_path = ingestion_obj.initiate_data_ingestion()

    ## Step 2:Transform Data
    transformation_obj = DataTransformation()
    scaled_train_arr, scaled_test_arr, unscaled_train_arr,unscaled_test_arr, preprocessor = transformation_obj.initiate_data_transformation(train_path, test_path)

    ## Step 3:Train Models
    model_trainer = ModelTrainer()
    model_name, score, cm, report = model_trainer.initiate_model_trainer(
        scaled_train_array = scaled_test_arr,
        scaled_test_array = scaled_test_arr,
        unscaled_train_array = unscaled_train_arr,
        unscaled_test_array = unscaled_test_arr,
        preprocessor= preprocessor
        )
    print(f"Best Model: {model_name}\n"
          f"F1 score: {score:.4f}\n" 
          f"False Negatives (class 0 as positive): {cm[0][1]}\n"
          f" Confusion matrix:\n {cm}\n"
          f"Classification Report:\n {report}")