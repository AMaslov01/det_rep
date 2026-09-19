FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app
COPY requirements-gateway.txt ./
RUN pip install --no-cache-dir -r requirements-gateway.txt
COPY gemini_gateway ./gemini_gateway

CMD ["sh", "-c", "exec uvicorn gemini_gateway.app:app --host 0.0.0.0 --port ${PORT}"]
