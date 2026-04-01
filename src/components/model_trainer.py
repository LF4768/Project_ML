import os 
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression,Ridge,ElasticNet,Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,root_mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from catboost import CatBoostRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path:str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "K-Neighbor Regressor": KNeighborsRegressor(),
                "Gradient Boost": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "Xg Boost": XGBRegressor(),
                "Cat Boost": CatBoostRegressor(verbose=False),
                "SVR": SVR(),
                "Linear Regression": LinearRegression(),
                "ElasticNet": ElasticNet(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "Logistic Regression": LogisticRegression(max_iter=500)
            }

            params={
                "Random Forest": {
                    "n_estimators" : [100,200,500,1000],
                    # "criterion" : ["squared_error", "absolute_error","friedman_mse","poisson"],
                    # "max_depth":[None,2,5,10],
                },
                "Decision Tree": {
                    "criterion": ["squared_error","friedman_mse","absolute_error","poisson"],
                    # "max_depth": [None,2,5,10],
                    # "min_samples_split": [2,3,4,5,10]
                },
                "K-Neighbor Regressor": {
                    "n_neighbors": [3,5,8,10,12],
                    # "algorithm": ["auto","ball_tree","kd_tree","brute"]
                },
                "Gradient Boost": {
                    # "loss": ["squared_error","absolute_error","huber","quantile"],
                    # "learning_rate": [0.1,0.01,0.001],
                    "n_estimators": [100,200,300,500,1000],
                    # "max_depth": [1,3,4,5,10],   
                },
                "AdaBoost": {
                    "n_estimators": [50,100,200,400],
                    "learning_rate": [1,0.1,0.01,0.001],
                    # "loss":["linear","square","exponential"]
                },
                "Xg Boost": {
                    "learning_rate": [0.1,0.5,0.01,0.7],
                    "n_estimators":[8,16,32,64,128,256],
                    # "max_depth": [4,5,6,10],
                    # "colsample_bytree":[0.1,0.01,0.0001,1]
                },
                "Cat Boost": {
                    # "iterations": [100,200,500],
                    "learning_rate": [0.1,0.01,0.001,0.3],
                    # "depth":[2,4,5,6,8,10]
                },
                "SVR": {
                    # "kernal": ["linear","poly","rbf","sigmoid","precomputed"],
                    # "degree":[1,2,3,4,5],
                    "epsilon":[0.1,0.01,0.001],
                },
                "Linear Regression": {},
                "ElasticNet": {},
                "Ridge": {},
                "Lasso": {},
                "Logistic Regression": {}
            }


            model_report:dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No Best Model Found")

            logging.info(f"Best model is: {best_model}")

            save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=best_model)

            # predicted = best_model.predict(X_test)
            return best_model_score

        except Exception as e:
            raise CustomException(e,sys)