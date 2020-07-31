from fastapi import FastAPI
import uvicorn
from typing import Dict
from pydantic import BaseModel
import json
import pandas as pd
import numpy as np
import joblib
import pickle
import tensorflow as tf
from geopy.geocoders import Nominatim

app = FastAPI()

# domain where this api is hosted for example : localhost:5000/docs to see swagger documentation automagically generated.


class Pred(BaseModel):
    roomtype: str
    accomodates: int
    bathrooms: float
    city: str
    latitude: float
    longitude: float
    reviewscoresrating: float
    bedrooms: float
    beds: str
    tv: int
    streetaddress: str
    zipcode: int


@app.get("/")
def home():
    return {"message": "Hello! got to https://testapifortesting.herokuapp.com/docs "}


@app.post('/predict')
def predict(pred: Pred):
    """This wil use Unit 3 model!"""
    test_model_data = pd.read_csv('models/sample_pred.csv')
    test_model_data.room_type = pred.roomtype
    test_model_data.accomodates = pred.accomodates
    test_model_data.bathrooms = pred.bathrooms
    test_model_data.zipcode = pred.zipcode
    test_model_data.bedrooms = pred.bedrooms

    model = joblib.load('models/prediction.pkl')
    response = round(model.predict(test_model_data)[0], 2)
    response = np.expm1(response)
    price_list = ['Prices']
    response_list = [round(response, 2)]
    response_dict = dict(zip(price_list, response_list))
    response_json = json.dumps(response_dict)
    return f'{str(response_json)}'


@app.post('/predict2')
def predict2(pred: Pred):
    '''This will use Unit 4 model!'''

    # This will call the test rows
    test_model_data = pd.read_csv('test_model.csv')
    # Editing test rows
    if pred.roomtype == 'Entire home/apt':
        test_model_data.room_type = 1.0
    elif pred.roomtype == 'Private room':
        test_model_data.room_type = 2.0
    else:
        test_model_data.room_type = 1.0

    # Applying
    test_model_data.accomodates = pred.accomodates
    test_model_data.bathrooms = pred.bathrooms
    test_model_data.latitude = pred.latitude
    test_model_data.longitude = pred.longitude
    test_model_data.zipcode = 0.0
    test_model_data.review_scores_rating = pred.reviewscoresrating
    test_model_data.bedrooms = pred.bedrooms
    test_model_data.TV = pred.tv

    # This will geocode
    ''' This will geocode addresses. And will set a dummy data if addrest do not work! '''
    geolocator = Nominatim(user_agent="Test")

    try:
        location = geolocator.geocode(f"{pred.streetaddress} , {pred.zipcode}")
        test_model_data.zipcode = 0.0
        test_model_data.latitude = location.latitude
        test_model_data.longitude = location.longitude
    except:
        pass

    # This is where the model is used.
    model = tf.keras.models.load_model('kristine_model_1')
    response = round(model.predict(test_model_data)[[0]][0][0], 2)
    response = np.expm1(response)
    price_list = ['Prices']
    response_list = [str(round(response, 2))]
    response_dict = dict(zip(price_list, response_list))
    response_json = json.dumps(response_dict)

    return f'{str(response_json)}'
