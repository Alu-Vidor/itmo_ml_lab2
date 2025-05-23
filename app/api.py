from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import shutil
import os
import uuid

from .model import predict
from .utils import read_imagefile, preprocess_image, decode_prediction

app = FastAPI()

# Папка для шаблонов и статики
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

UPLOAD_DIR = "app/static"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/", response_class=HTMLResponse)
async def classify(request: Request, file: UploadFile = File(...)):
    # Читаем файл
    contents = await file.read()
    image = read_imagefile(contents)
    img_array = preprocess_image(image)
    prediction = predict(img_array)
    label, confidence = decode_prediction(prediction)

    # Сохраняем картинку для предпросмотра (уникальное имя)
    img_id = str(uuid.uuid4())[:8]
    img_path = f"{UPLOAD_DIR}/upload_{img_id}.jpg"
    image.save(img_path)

    result = {
        "label": label,
        "confidence": confidence,
        "image_url": f"/static/upload_{img_id}.jpg"
    }
    return templates.TemplateResponse("index.html", {"request": request, "result": result})

# API endpoint для запросов через POSTman/HTTP
@app.post("/api/predict/")
async def api_predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = read_imagefile(contents)
    img_array = preprocess_image(image)
    prediction = predict(img_array)
    label, confidence = decode_prediction(prediction)
    return {"label": label, "confidence": confidence}