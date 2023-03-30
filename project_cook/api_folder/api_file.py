from fastapi import Body, FastAPI, File, UploadFile
from io import BytesIO

from project_cook.interface.main import pred_streamlit, search_recipes, transcribe_audio
import os
from tempfile import NamedTemporaryFile
import aiofiles
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable

from fastapi import UploadFile


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
    return pred_streamlit(img, lang)

@api.post("/what-you-say/{lang}")
async def what_you_say(lang: str, uploaded_file: UploadFile=File(...)):
    file_location = f"./notebooks/audio/{uploaded_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(uploaded_file.file.read())
    return transcribe_audio(file_location, lang)
