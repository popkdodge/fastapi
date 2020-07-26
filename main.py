from fastapi import FastAPI
import uvicorn
from typing import Dict
from pydantic import BaseModel


app = FastAPI()

#domain where this api is hosted for example : localhost:5000/docs to see swagger documentation automagically generated.

@app.get("/")
def home():
    return {"message":"Hello TutLinks.com"}

@app.post('/predict')
def predict(pred: Pred):
    import pandas as pd
    import joblib
    import pickle
    test_model_data = pd.read_csv('models/sample_pred.csv')
    model = joblib.load('models/prediction.pkl')
    response  = round(model.predict(test_model_data)[0],2)
    price_list = ['Prices']
    response_list = [f'{round(response,2)}']
    response_dict = dict(zip(price_list, response_list))
    response_json = json.dumps(response_dict)
    return response_json

