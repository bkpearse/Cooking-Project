from fastapi import Body, FastAPI, File, UploadFile

from project_cook.interface.main import (pred_streamlit, predict_function,
                                         search_recipes)

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


@api.get("/query-recipes/{lang}")
async def what_to_eat(lang: str, ingredients):
    return search_recipes(ingredients, lang)


@api.post("/what-to-eat/{lang}")
async def what_to_eat(lang: str, img: UploadFile = File(...)):
    img = img.file.read()
    predicted_class = pred_streamlit(img, lang)
    return predicted_class
