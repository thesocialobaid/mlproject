import os
import sys
from dataclasses import dataclass

import mlflow
import mlflow.sklearn
import mlflow.catboost                          # ← add this

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error  # ← updated
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model


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
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
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
                    "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                },
                "Random Forest": {"n_estimators": [8, 16, 32, 64, 128, 256]},
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Linear Regression": {},
                "K-Neighbors Regressor": {"n_neighbors": [5, 7, 9, 11]},
                "XGB Regressor": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "CatBoosting Regressor": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100],
                },
                "AdaBoost Regressor": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
            }

            # ── Step 1: evaluate all models ──
            model_report: dict = evaluate_model(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                models=models, params=params,
            )

            # ── Step 2: configure MLflow ──         ← UNCOMMENTED
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
            mlflow.set_experiment("student-exam-performance")

            # ── Step 3: log every model as its own run ──
            logging.info("Logging all model results to MLflow")

            for model_name, model_score in model_report.items():
                with mlflow.start_run(run_name=model_name):
                    mlflow.log_params(models[model_name].get_params())

                    preds = models[model_name].predict(X_test)
                    mlflow.log_metric("r2_score", model_score)
                    mlflow.log_metric("mae", mean_absolute_error(y_test, preds))
                    mlflow.log_metric("rmse", root_mean_squared_error(y_test, preds))  # ← fixed

                    mlflow.set_tag("model_name", model_name)

                    # ── Route to correct MLflow flavor ──        ← fixed
                    if isinstance(models[model_name], CatBoostRegressor):
                        mlflow.catboost.log_model(models[model_name], artifact_path="model")
                    else:
                        mlflow.sklearn.log_model(models[model_name], artifact_path="model")

                    logging.info(f"MLflow logged → {model_name} | R2: {model_score:.4f}")

            # ── Step 4: pick the best model ──
            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No Best Model found", sys)

            logging.info(f"Best model: {best_model_name} | R2: {best_model_score:.4f}")

            # ── Step 5: log the winner separately ──
            with mlflow.start_run(run_name=f"BEST__{best_model_name}"):
                mlflow.log_params(best_model.get_params())

                best_preds = best_model.predict(X_test)
                mlflow.log_metric("r2_score", best_model_score)
                mlflow.log_metric("mae", mean_absolute_error(y_test, best_preds))
                mlflow.log_metric("rmse", root_mean_squared_error(y_test, best_preds))  # ← fixed

                mlflow.set_tag("model_name", best_model_name)
                mlflow.set_tag("status", "best")

                if isinstance(best_model, CatBoostRegressor):
                    mlflow.catboost.log_model(best_model, artifact_path="model")
                else:
                    mlflow.sklearn.log_model(best_model, artifact_path="model")

            # ── Step 6: save pkl ──
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            predicted = best_model.predict(X_test)
            return r2_score(y_test, predicted)

        except Exception as e:
            raise CustomException(e, sys)