import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "app/model.h5"
model = load_model(MODEL_PATH)

def predict(img_array: np.ndarray) -> np.ndarray:
    """
    Принимает подготовленный массив изображения (batch, height, width, channels).
    Возвращает массив предсказаний (batch_size, 1).
    """
    preds = model.predict(img_array)
    return preds