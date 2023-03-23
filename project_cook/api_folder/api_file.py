from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile
from project_cook.interface.whoosh_search import search_recipes
from project_cook.interface.images_predict import predict_function, pred_streamlit
import aiofiles
from pathlib import Path

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

    return {'prediction': int(feature1) * int(feature2)}


# @api.post("/what-to-eat")
# async def what_to_eat(file: UploadFile = File(...)):
#     filepath = 'notebooks/images/' + file.filename[:-4]
#     results = []
#     async with aiofiles.open(filepath+'/'+file.filename, 'wb') as out_file:
#         content = await file.read()  # async read
#         await out_file.write(content)  # async write
#         results = predict_function(Path(filepath))
#     return {"Result": results}
#     # return {'file': file, 'content': contents}


@api.get("/query-recipes")
async def what_to_eat(ingredients):
    return search_recipes(ingredients)


@api.post("/what-to-eat")
async def what_to_eat(img: UploadFile = File(...)):
    print(img.filename)
    extension = img.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    # return img
    # if not extension:
    #     return f"{img.filename} Image must be .jpg, .jpeg or .png format!"
    img = img.file.read()
    predicted_class = pred_streamlit(img)

    return predicted_class
