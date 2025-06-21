FROM python:3.11-slim

# Installiere Tesseract + andere Abhängigkeiten
RUN apt-get update && \
    apt-get install -y tesseract-ocr libglib2.0-0 libsm6 libxext6 libxrender-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
    
RUN pip install tflite-runtime

# Arbeitsverzeichnis
WORKDIR /app

# Kopiere Abhängigkeiten und App
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]

