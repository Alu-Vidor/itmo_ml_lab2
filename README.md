# itmo_ml_lab2 — Классификация свежести фруктов и овощей

Веб-приложение для определения свежести фруктов и овощей по фотографии. Использует сверточную нейросеть VGG16 для бинарной классификации: "свежий" или "испорченный" продукт.

---

## 📦 Быстрый старт (Docker)

> **Требования:**  
> - Docker и Docker Compose должны быть установлены на вашей машине.

### 1. Клонируйте репозиторий и перейдите в папку проекта

```bash
git clone <ваш-репозиторий>
cd itmo_ml_lab2/docker
```

### 2. Соберите и запустите сервис

```bash
docker compose up --build
```

### 3. Откройте приложение

- Перейдите в браузере по адресу:  
  [http://localhost:8000](http://localhost:8000)

---

## 🛠️ Структура проекта

```
itmo_ml_lab2/
├── app/
│   ├── api.py             # FastAPI сервер и маршруты
│   ├── model.py           # Загрузка и инференс модели
│   ├── utils.py           # Функции обработки изображений
│   ├── templates/
│   │   └── index.html     # Jinja2-шаблон главной страницы
│   ├── static/
│   └── model.h5           # Файл обученной модели
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yaml
├── requirements.txt
└── README.md
```

---

## 🌐 API-эндпоинты

### Веб-интерфейс

- **Главная страница:**  
  `GET /` — форма загрузки изображения и вывод результата.

### Программный доступ (API)

- **POST /api/predict/**  
  Принимает файл изображения (`multipart/form-data`).  
  **Ответ:**
  ```json
  {
    "label": "Fresh Fruit",
    "confidence": 0.975
  }
  ```
  - `label` — результат ("Fresh Fruit" или "Rotten Fruit")
  - `confidence` — вероятность (от 0 до 1)

- **Swagger UI:**  
  [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📝 Лицензия

[MIT License](LICENSE)

---
