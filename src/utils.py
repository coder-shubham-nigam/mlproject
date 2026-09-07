import os 
import sys
import numpy as np 
import pandas as pd 
import dill # type: ignore
from src.exception import CustomException
from sklearn.metrics import r2_score # type: ignore
from sklearn.model_selection import GridSearchCV # type: ignore

def save_object(file_path , obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path , exist_ok=True)
        
        with open(file_path , 'wb') as f:
            dill.dump(obj,f)
    
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(xtrain , ytrain , xtest , ytest, models , params):
    try:        
        report = {}

        for model_name, model in models.items():
            para = params[model_name]
            
            gs = GridSearchCV(model, para , cv = 3)
            gs.fit(xtrain,ytrain)
            
            model.set_params(**gs.best_params_)
            model.fit(xtrain, ytrain) # Train model

            # Make predictions
            y_train_pred = model.predict(xtrain)
            y_test_pred = model.predict(xtest)
            
            train_model_score = r2_score(ytrain , y_train_pred)
            test_model_score = r2_score(ytest , y_test_pred)
            
            report[model_name] = test_model_score
            
        return report
    except Exception as e:
        raise CustomException(e,sys)

def load_object(file_path):
    try:
        with open(file_path , 'rb') as f:
            return dill.load(f)
    except Exception as e:
        raise CustomException(e,sys)