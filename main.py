from fastapi import FastAPI
import uvicorn
from typing import Dict
from pydantic import BaseModel
import json


app = FastAPI()

#domain where this api is hosted for example : localhost:5000/docs to see swagger documentation automagically generated.
class Pred(BaseModel):
    propertytype: str
    roomtype: str
    accomodates: int
    bathrooms: float
    cleanfee: bool
    city: str
    latitude: float
    longitude: float
    reviewscoresrating: float
    zipcode: int
    bedrooms: float
    beds: float
    dryer: bool
    parking: bool
    descriptionlen: int


@app.get("/")
def home():
    return {"message":"Hello! got to https://testapifortesting.herokuapp.com/docs "}

@app.post('/predict')
def predict(pred: Pred):
    import pandas as pd
    import numpy as np
    import joblib
    import pickle
    test_model_data = pd.read_csv('models/sample_pred.csv')
    test_model_data.property_type = pred.propertytype
    test_model_data.room_type = pred.roomtype
    test_model_data.accomodates = pred.accomodates
    test_model_data.bathrooms = pred.bathrooms
    test_model_data.clean_fee = pred.cleanfee
    test_model_data.city = pred.city
    test_model_data.zipcode = pred.zipcode
    test_model_data.bedrooms = pred.bedrooms
    test_model_data.Description_Len = pred.descriptionlen

    model = joblib.load('models/prediction.pkl')
    response  = round(model.predict(test_model_data)[0],2)
    response = np.expm1(response)
    price_list = ['Prices']
    response_list = [round(response,2)]
    response_dict = dict(zip(price_list, response_list))
    response_json = json.dumps(response_dict)
    return f'{str(response_json)}'

