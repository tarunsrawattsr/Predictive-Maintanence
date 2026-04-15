FROM python3.11-slim

WORKDIR /app

COPY requirement.txt .

RUN pip install --no-cache-dir -r requirement.txt

COPY . .

EXPOSE 8000 8501

CMD ["python3", "train.py"]
