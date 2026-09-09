FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -e .
CMD ["aegisformer", "profile", "--config", "configs/baseline.yaml"]
