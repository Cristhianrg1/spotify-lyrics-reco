# Dockerfile

FROM python:3.11-slim

# Evitar archivos .pyc y usar stdout sin buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Directorio de trabajo
WORKDIR /app

# Copiamos todo el proyecto
COPY . .

# Instalamos dependencias del proyecto (usa pyproject.toml)
# Si tu pyproject está bien definido, esto instala fastapi, uvicorn, google-cloud-bigquery, motor, httpx, etc.
RUN pip install --no-cache-dir .

# Cloud Run define la variable PORT; usamos 8080 por defecto para local
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]