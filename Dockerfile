FROM python:3.9-slim

RUN apt-get update && apt-get install -y ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5001

CMD ["gunicorn", "server:app", "--bind", "0.0.0.0:5001", "--timeout", "120", "--workers", "2", "--threads", "2"]