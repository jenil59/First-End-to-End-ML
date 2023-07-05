import os
import sys
import dill
from .logger import logging
import numpy as np
import pandas as pd

from src.exception import CustomException

from sklearn.metrics import r2_score
from  sklearn.model_selection import GridSearchCV


def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as f:
            dill.dump(obj,f)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(Xtrain,Ytrain,Xtest,Ytest,models,param_grid=None):
    try:
        logging.info("Model Training with tuning is started .. ")

        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param_grid[list(models.keys())[i]]

            gs = GridSearchCV(model,param_grid=para,cv=3,n_jobs=-1)
            gs.fit(Xtrain,Ytrain)



            model.set_params(**gs.best_params_)
            model.fit(Xtrain,Ytrain)

            y_train_pred = model.predict(Xtrain)

            y_test_pred = model.predict(Xtest)

            train_model_score = r2_score(y_true=Ytrain,y_pred=y_train_pred)
            test_model_score = r2_score(y_true=Ytest,y_pred=y_test_pred)

            report[list(models.keys())[i]] = test_model_score

            logging.info("Model Training with tuning is completed .. ")
        return report

    except Exception as e:
        raise CustomException(e,sys)