# 🎵 Spotify Lyrics Recommendation System

Un sistema para buscar y recomendar letras de canciones de Spotify utilizando búsqueda semántica con FastAPI, MongoDB y Docker.

## 🚀 Características Principales

- Extracción de metadatos de álbumes y canciones de Spotify
- Almacenamiento de letras con búsqueda semántica usando MongoDB
- API REST con FastAPI para ingesta y búsqueda de letras
- Despliegue contenerizado con Docker
- Gestión de dependencias con `uv` para un rendimiento óptimo

## 🛠️ Requisitos Previos

- Docker instalado
- Cuenta de desarrollador de Spotify (para credenciales de API)
- MongoDB Atlas (para búsqueda vectorial)
- Google Cloud SDK (opcional, para BigQuery)

## 🚀 Empezando Rápido

### Usando Docker

1. Clona el repositorio:
   ```bash
   git clone <repo-url>
   cd spotify-lyrics-reco
   ```

2. Configura las variables de entorno:
   ```bash
   cp .env-example .env
   # Edita el archivo .env con tus credenciales de Spotify y MongoDB
   ```

3. Construye la imagen de Docker:
   ```bash
   docker build -t spotify-lyrics-reco .
   ```

4. Ejecuta el contenedor:
   ```bash
   docker run -d --name spotify-lyrics -p 8000:8000 --env-file .env spotify-lyrics-reco
   ```

5. La API estará disponible en `http://localhost:8000`
   - Documentación interactiva: `http://localhost:8000/docs`
   - Alternativa: `http://localhost:8000/redoc`

### Desarrollo Local

1. Crea y activa un entorno virtual:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # En Windows: .venv\Scripts\activate
   ```

2. Instala `uv` para un manejo rápido de dependencias:
   ```bash
   curl -sSfL https://astral.sh/uv/install.sh | sh
   ```

3. Instala dependencias con `uv`:
   ```bash
   uv pip install -e .
   ```

4. Inicia el servidor de desarrollo:
   ```bash
   uvicorn main:app --reload
   ```

## 📚 Documentación de la API

La documentación interactiva de la API está disponible en:
- Swagger UI: `http://localhost:8080/docs`
- ReDoc: `http://localhost:8080/redoc`

### Endpoints Principales

#### Ingresar un álbum
```http
POST /ingest-album
Content-Type: application/json

{
  "album_ref": "spotify:album:ID_DEL_ALBUM"
}
```

#### Buscar letras
```http
POST /search-lyrics
Content-Type: application/json

{
  "query": "tu búsqueda aquí",
  "top_k": 5
}
```

## 🏗️ Estructura del Proyecto

```
spotify-lyrics-reco/
├── src/                    # Código fuente
│   ├── pipelines/         # Pipelines de ingesta de datos
│   ├── services/          # Lógica de negocio
│   └── clients/           # Clientes para servicios externos
├── .env-example          # Plantilla de variables de entorno
├── Dockerfile            # Configuración de la imagen de la aplicación
└── pyproject.toml        # Configuración del proyecto y dependencias
```

## 🔧 Variables de Entorno

Crea un archivo `.env` basado en `.env-example` con las siguientes variables:

```
# Spotify
SPOTIFY_CLIENT_ID=tu_client_id
SPOTIFY_CLIENT_SECRET=tu_client_secret

# MongoDB
MONGODB_URI=tu_mongodb_uri

# Google Cloud (opcional)
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
```

## 🐛 Solución de Problemas

### Problemas Comunes

#### Error de conexión a MongoDB
- Verifica que `MONGODB_URI` en `.env` sea correcta
- Asegúrate de que tu IP esté en la lista blanca de MongoDB Atlas

#### Credenciales de Spotify inválidas
- Verifica `SPOTIFY_CLIENT_ID` y `SPOTIFY_CLIENT_SECRET` en `.env`
- Asegúrate de que las credenciales sean válidas y tengan los permisos necesarios

#### Problemas con Docker
- Reconstruye la imagen si hay cambios:
  ```bash
  docker build -t spotify-lyrics-reco .
  docker stop spotify-lyrics || true
  docker rm spotify-lyrics || true
  docker run -d --name spotify-lyrics -p 8000:8000 --env-file .env spotify-lyrics-reco
  ```
- Verifica los logs del contenedor:
  ```bash
  docker logs -f spotify-lyrics
  ```
- Limpia recursos no utilizados:
  ```bash
  docker system prune
  ```

### Reiniciar la Aplicación
Para reiniciar la aplicación después de hacer cambios:
```bash
docker stop spotify-lyrics
docker start spotify-lyrics
```

O para reconstruir completamente:
```bash
docker build -t spotify-lyrics-reco .
docker stop spotify-lyrics || true
docker rm spotify-lyrics || true
docker run -d --name spotify-lyrics -p 8000:8000 --env-file .env spotify-lyrics-reco
```