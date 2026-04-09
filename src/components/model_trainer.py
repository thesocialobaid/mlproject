# what catboost regressor brings to the table? 
# Handles categorical features natively - no need for manual label encoding or one-hot encoding 
# Less hyperparameter tuning needed - works well out of the box
# Resistant to overfitting - uses ordered boosting internally 
# Fast training - especially on datasets 

import os 
import sys 
from dataclasses import dataclass                       

import mlflow
import mlflow.sklearn

from catboost import CatBoostRegressor 
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression        
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException 
from src.logger import logging 

from src.utils import save_object 
from src.utils import evaluate_model

@dataclass 
class ModelTrainerConfig: 
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    
class ModelTrainer: 
    def __init__(self): 
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array, preprocessor_path): 
        try: 
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (        
                   train_array[:,:-1], 
                   train_array[:,-1],
                   test_array[:,:-1],
                   test_array[:,-1]
            )

            models = {
                "Random Forest": RandomForestRegressor(), 
                "Decision Tree": DecisionTreeRegressor(), 
                "Gradient Boosting": GradientBoostingRegressor(), 
                "Linear Regression": LinearRegression(), 
                "K-Neighbors Regressor": KNeighborsRegressor(), 
                "XGB Regressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False), 
                "AdaBoost Regressor": AdaBoostRegressor(), 
            }
            params = {
                "Decision Tree": { 
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                }, 
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {}, 
                "K-Neighbors Regressor": { 
                    'n_neighbors': [5, 7, 9, 11], 
                }, 
                "XGB Regressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            } 

            # ── MLflow: set where runs are stored and name the experiment ──
            mlflow.set_tracking_uri("mlruns")          # saves inside your project root
            mlflow.set_experiment("student-exam-performance")

            # ── evaluate all models (your existing logic, untouched) ──
            model_report: dict = evaluate_model(
                X_train=X_train, y_train=y_train, 
                X_test=X_test, y_test=y_test, 
                models=models,
                params=params
            )
            
            # ── log every model as its own MLflow run ──
            for model_name, model_score in model_report.items():
                with mlflow.start_run(run_name=model_name):
                    
                    # log the best hyperparams found for this model
                    best_params = models[model_name].get_params()
                    mlflow.log_params(best_params)

                    # log the r2 score from evaluate_model
                    mlflow.log_metric("r2_score", model_score)

                    # re-predict with this model to get MAE and RMSE too
                    preds = models[model_name].predict(X_test)
                    mlflow.log_metric("mae",  mean_absolute_error(y_test, preds))
                    mlflow.log_metric("rmse", mean_squared_error(y_test, preds, squared=False))

                    # tag so you can filter in the UI
                    mlflow.set_tag("model_name", model_name)

                    # log the model artifact itself
                    mlflow.sklearn.log_model(models[model_name], artifact_path="model")

                    logging.info(f"MLflow run logged → {model_name} | R2: {model_score:.4f}")

            # ── pick the best model (your existing logic, untouched) ──
            best_model_score = max(sorted(model_report.values()))
            best_model_name  = list(model_report.keys())[ 
                list(model_report.values()).index(best_model_score)                           
            ]
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:                   
                raise CustomException("No Best Model found", sys)

            logging.info(f"Best model: {best_model_name} with R2 score: {best_model_score:.4f}")

            # ── log the winner separately so it's easy to find in the UI ──
            with mlflow.start_run(run_name=f"BEST__{best_model_name}"):
                mlflow.log_params(best_model.get_params())
                mlflow.log_metric("r2_score", best_model_score)

                best_preds = best_model.predict(X_test)
                mlflow.log_metric("mae",  mean_absolute_error(y_test, best_preds))
                mlflow.log_metric("rmse", mean_squared_error(y_test, best_preds, squared=False))

                mlflow.set_tag("model_name", best_model_name)
                mlflow.set_tag("status", "best")          # easy to filter in UI
                mlflow.sklearn.log_model(best_model, artifact_path="model")
                
            # ── save pkl as before (your existing logic, untouched) ──
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
                
            predicted = best_model.predict(X_test)
            score = r2_score(y_test, predicted)
            return score

        except Exception as e: 
            raise CustomException(e, sys)