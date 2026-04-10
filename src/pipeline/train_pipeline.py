# src/pipeline/train_pipeline.py

import sys
from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            # Step 1: Data Ingestion
            logging.info("Starting data ingestion")
            data_ingestion = DataIngestion()
            train_path, test_path = data_ingestion.initiate_data_ingestion()

            # Step 2: Data Transformation
            logging.info("Starting data transformation")
            data_transformation = DataTransformation()
            train_array, test_array, preprocessor_path = data_transformation.initiate_data_transformation(
                train_path, test_path
            )

            # Step 3: Model Training + MLflow logging
            logging.info("Starting model training")
            model_trainer = ModelTrainer()
            r2_score = model_trainer.initiate_model_trainer(
                train_array, test_array, preprocessor_path
            )

            logging.info(f"Training complete. Best model R2 Score: {r2_score:.4f}")
            return r2_score

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainPipeline()
    score = pipeline.run_pipeline()
    print(f"\nFinal R2 Score: {score:.4f}")