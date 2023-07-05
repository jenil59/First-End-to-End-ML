import os
import sys
from dataclasses import dataclass

import numpy as np
from hyperopt import hp

from catboost import CatBoostRegressor

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('split training and test input data')

            X_train,y_train = train_array[:,:-1],train_array[:,-1]
            X_test,y_test = test_array[:,:-1],test_array[:,-1]

            models = {
                'Decision Tree': DecisionTreeRegressor(),
                'K-Neighbors ': KNeighborsRegressor(),
                'Ridge':Ridge(),
                'lasso':Lasso(),
                'Random Forest':RandomForestRegressor(),
                'Gradient Boosting':GradientBoostingRegressor(),
                'XGBR':XGBRegressor(),
                'Catboost':CatBoostRegressor(verbose=False),
                'Adaboost':AdaBoostRegressor()
            }

            # hyper parameter tuning
            params = {
                'Decision Tree': {
                    'criterion' : ['squared_error','friedman_mse','absolute_error','poisson'],
                    #'splitter':['best','random'],
                    'max_features':['sqrt','log2']
                },
                'K-Neighbors ': {
                        'n_neighbors' : [5,7,9,11,13,15],
                        'weights' : ['uniform','distance'],
                        'metric' : ['minkowski','euclidean','manhattan']
                },
                'Ridge':{
                    'alpha':[1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]
                },
                'lasso':{
                    'alpha':[1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]
                },
                'Random Forest':{
                    'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                    'min_samples_leaf': [1, 2, 4],
                    'min_samples_split': [2, 5, 10],
                    'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
                },
                'Gradient Boosting':{
                    "n_estimators":[5,50,250,500],
                    "max_depth":[1,3,5,7,9],
                    "learning_rate":[0.01,0.1,1,10,100]
                },
                'XGBR':{
                    'learning_rate' : [0.1,0.01,0.05,1],
                    'n_estimators':[8,16,32,64,128,256],
                },
                'Catboost':{
                    'depth':[6,8,10],
                    'learning_rate':[0.01,0.05,0.1],
                    'iterations':[30,50,100]
                },
                'Adaboost':{
                    'learning_rate':[0.1,0.01,0.05,.001],
                    'loss':['linear','square','exponential'],
                    'n_estimators':[8,16,32,64,128,256]

                }
            }
            
            model_report:dict = evaluate_models(Xtrain=X_train,Ytrain=y_train,Xtest=X_test,Ytest=y_test,models=models,param_grid = params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException('No Best Model Found')
            
            logging.info('Best Model is Found for training and testing')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test,predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e,sys)
