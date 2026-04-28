FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    ultralytics==8.2.0 \
    opencv-python-headless==4.9.0.80 \
    azure-eventhub==5.11.7 \
    azure-storage-blob==12.19.1 \
    azure-identity==1.16.0

COPY app.py .

ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]
