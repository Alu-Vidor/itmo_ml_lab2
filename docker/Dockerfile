FROM python:3.11-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем requirements.txt
COPY requirements.txt .

# Устанавливаем Python-зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем всё приложение
COPY ./app ./app

# Открываем порт для FastAPI
EXPOSE 8000

# Запуск приложения
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]