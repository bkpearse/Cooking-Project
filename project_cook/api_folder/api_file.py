from fastapi import Body, FastAPI, File, UploadFile

from project_cook.interface.main import pred_streamlit, search_recipes

api = FastAPI()


# define a root `/` endpoint
@api.get("/")
def index():
    return {"ok": "API connected"}


@api.get("/query-recipes/{lang}")
async def what_to_eat(lang: str, ingredients):
    return search_recipes(ingredients, lang)


@api.post("/what-to-eat/{lang}")
async def what_to_eat(lang: str, img: UploadFile = File(...)):
    img = img.file.read()
    predicted_class = pred_streamlit(img, lang)
    return predicted_class
