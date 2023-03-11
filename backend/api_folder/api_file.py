from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile
from PIL import Image
import numpy as np

api = FastAPI()

# define a root `/` endpoint
@api.get("/")
def index():
    return {"ok": "API connected"}


@api.get("/predict")
def predict(feature1, feature2):

    # model = picle.load_model()
    # prediction = model.predict(feature1, feature2)

    # Here, I'm only returning the features, since I don't actually have a model.
    # In a real life setting, you would return the predictions.

    return {'prediction': int(feature1)*int(feature2)}


@api.post("/what-to-eat")
async def what_to_eat(file: UploadFile = File(...)):
    # TODO return recipes from photo.
    return {'file': file}
