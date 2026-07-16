FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train the model once at build time so the image is ready to run immediately.
RUN python src/preprocess.py && python src/train.py

EXPOSE 8501

CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0"]
