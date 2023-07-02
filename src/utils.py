import os
import sys
import dill

import numpy as np
import pandas as pd

from src.exception import CustomException

from sklearn.metrics import r2_score


def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as f:
            dill.dump(obj,f)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(Xtrain,Ytrain,Xtest,Ytest,models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            model.fit(Xtrain,Ytrain)

            y_train_pred = model.predict(Xtrain)

            y_test_pred = model.predict(Xtest)

            train_model_score = r2_score(y_true=Ytrain,y_pred=y_train_pred)
            test_model_score = r2_score(y_true=Ytest,y_pred=y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e,sys)