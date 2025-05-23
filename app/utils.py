import numpy as np
from PIL import Image
import io

def read_imagefile(file) -> Image.Image:
    """
    Чтение загруженного через API файла (bytes) и конвертация в PIL Image.
    """
    image = Image.open(io.BytesIO(file)).convert("RGB")
    return image

def preprocess_image(image: Image.Image, target_size=(150, 150)) -> np.ndarray:
    """
    Масштабирование, преобразование к numpy-массиву, нормализация, расширение размерности для подачи в модель.
    """
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0   # нормализация
    img_array = np.expand_dims(img_array, axis=0)  # batch axis
    return img_array

def decode_prediction(pred: np.ndarray, threshold: float = 0.5):
    """
    Декодирует результат модели в человекочитаемый формат и возвращает label и confidence.
    """
    confidence = float(pred[0][0])
    if confidence > threshold:
        return "Rotten Fruit", confidence
    else:
        return "Fresh Fruit", 1.0 - confidence