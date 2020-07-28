from fastapi import FastAPI
import uvicorn
from typing import Dict
from pydantic import BaseModel
import json


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
    import pandas as pd
    import numpy as np
    import joblib
    import pickle
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
    import pandas as pd
    import numpy as np
    import joblib
    import pickle
    import tensorflow as tf

    test_model_data = pd.read_csv('test_model.csv')
#        test_model_data.room_type = 1.0
#    if pred.roomtype == 'Entire home/apt':
#    elif pred.roomtype == 'Private room':
#        test_model_data.room_type = 2.0
#    else:
#        test_model_data.room_type = 3.0
# j    test_model_data.zipcode = pred.zipcode
    model = tf.keras.models.load_model('kristine_model_0')
    response = round(model.predict(test_model_data)[[0]][0][0], 2)
    response = np.expm1(response)
    price_list = ['Prices']
    response_list = [str(round(response, 2))]
    response_dict = dict(zip(price_list, response_list))
    response_json = json.dumps(response_dict)
    return f'{str(response_json)}'
