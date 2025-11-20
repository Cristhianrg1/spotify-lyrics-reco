# Guía paso a paso: Docker + GCP + Cloud Run (macOS + Homebrew)

Este archivo está pensado para pegarlo tal cual en un `.md`.
Supone que ya tienes:
- Una cuenta en Google Cloud.
- Un proyecto creado (usaremos `PROJECT_ID` como placeholder).
- Un código fuente con un `Dockerfile` en la carpeta actual.

---

## 0. Variables base (ajusta a tu caso)

Antes de empezar, define estas variables en tu terminal (puedes pegarlas tal cual y editar):

```bash
export PROJECT_ID="TU_PROJECT_ID_AQUI"
export REGION="southamerica-east1"         # O la región que uses (ej: us-central1, us-east1)
export REPO_NAME="my-app-repo"             # Nombre del repo en Artifact Registry
export IMAGE_NAME="spotify-automation-app" # Nombre de la imagen
export SERVICE_NAME="spotify-automation-service" # Nombre del servicio en Cloud Run

# URI completo de la imagen en Artifact Registry
export IMAGE_URI="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME"
```

---

## 1. Instalar la CLI de Google Cloud con Homebrew (macOS)

```bash
# Actualizar Homebrew
brew update

# Instalar Google Cloud SDK
brew install --cask google-cloud-sdk

# Cargar la CLI en la sesión actual (a veces ya viene configurado)
if [ -f "$(brew --prefix)/Caskroom/google-cloud-sdk/latest/google-cloud-sdk/path.bash.inc" ]; then
  source "$(brew --prefix)/Caskroom/google-cloud-sdk/latest/google-cloud-sdk/path.bash.inc"
fi

if [ -f "$(brew --prefix)/Caskroom/google-cloud-sdk/latest/google-cloud-sdk/completion.bash.inc" ]; then
  source "$(brew --prefix)/Caskroom/google-cloud-sdk/latest/google-cloud-sdk/completion.bash.inc"
fi

# Verificar instalación
gcloud version
```

Si usas Zsh o Fish, revisa las instrucciones que imprime Homebrew tras la instalación para agregar `gcloud` al PATH permanentemente.

---

## 2. Autenticación y configuración inicial

### 2.1. Iniciar sesión

```bash
gcloud auth login
```

Se abrirá el navegador para que autorices la cuenta.

### 2.2. Seleccionar el proyecto

```bash
gcloud config set project "$PROJECT_ID"
```

### 2.3. Configurar región por defecto (para Cloud Run)

```bash
gcloud config set run/region "$REGION"
```

---

## 3. Habilitar APIs necesarias

```bash
# Cloud Run
gcloud services enable run.googleapis.com

# Artifact Registry (repositorio de imágenes)
gcloud services enable artifactregistry.googleapis.com

# Cloud Build (para construir imágenes desde el código)
gcloud services enable cloudbuild.googleapis.com
```

---

## 4. Crear repositorio de contenedores en Artifact Registry

```bash
gcloud artifacts repositories create "$REPO_NAME"   --repository-format=docker   --location="$REGION"   --description="Repositorio de imágenes Docker para mis servicios"
```

Para verificar que se creó correctamente:

```bash
gcloud artifacts repositories list --location="$REGION"
```

---

## 5. Construir y subir la imagen Docker (usando Cloud Build)

Opción recomendada: no necesitas Docker instalado localmente, solo `gcloud`.

Desde la carpeta donde está tu `Dockerfile`:

```bash
# Construir y subir la imagen a Artifact Registry
gcloud builds submit . --tag "$IMAGE_URI"
```

Cloud Build:
- Lee el `Dockerfile`.
- Construye la imagen.
- La sube automáticamente a Artifact Registry en la ruta `$IMAGE_URI`.

### (Opcional) Usar Docker local

Solo si ya tienes Docker instalado y corriendo.

```bash
# Construir imagen local
docker build -t "$IMAGE_NAME" .

# Etiquetar la imagen con el repo de Artifact Registry
docker tag "$IMAGE_NAME" "$IMAGE_URI"

# Autenticarse en Artifact Registry
gcloud auth configure-docker "$REGION-docker.pkg.dev"

# Subir imagen
docker push "$IMAGE_URI"
```

---

## 6. Deploy en Cloud Run

### 6.1. Desplegar el servicio por primera vez

```bash
gcloud run deploy "$SERVICE_NAME"   --image="$IMAGE_URI"   --region="$REGION"   --platform=managed   --memory=512Mi   --allow-unauthenticated
```

- `--allow-unauthenticated`: hace el servicio público (sin auth).
- Si quieres que requiera autenticación, elimina esa flag.

### 6.2. Obtener la URL del servicio

Al final del deploy verás algo como:

```text
Service [spotify-automation-service] revision [...] has been deployed and is serving traffic at https://<URL>.run.app
```

Si quieres consultar la URL después:

```bash
gcloud run services describe "$SERVICE_NAME"   --region="$REGION"   --format="value(status.url)"
```

---

## 7. Actualizar el servicio (nueva versión de la app)

Cada vez que cambies el código:

```bash
# 1) Construir y subir nueva imagen
gcloud builds submit . --tag "$IMAGE_URI"

# 2) Deployar nueva revisión del servicio
gcloud run deploy "$SERVICE_NAME"   --image="$IMAGE_URI"   --region="$REGION"   --platform=managed   --allow-unauthenticated
```

Cloud Run crea una nueva revisión y enruta el tráfico automáticamente a la nueva versión.

---

## 8. Variables de entorno (configs, secrets simples, etc.)

Para actualizar variables de entorno (ej: `ENVIRONMENT`, `DEBUG`):

```bash
gcloud run services update "$SERVICE_NAME"   --region="$REGION"   --update-env-vars="ENVIRONMENT=prod,DEBUG=false"
```

Para borrar una variable:

```bash
gcloud run services update "$SERVICE_NAME"   --region="$REGION"   --update-env-vars="DEBUG-"
```

Para añadir más variables:

```bash
gcloud run services update "$SERVICE_NAME"   --region="$REGION"   --update-env-vars="ENVIRONMENT=prod,DEBUG=false,API_KEY=foo"
```

---

## 9. Comandos útiles de operación

```bash
# Listar servicios de Cloud Run
gcloud run services list --region="$REGION"

# Ver detalles de un servicio específico
gcloud run services describe "$SERVICE_NAME" --region="$REGION"

# Ver últimas entradas de logs (Cloud Logging)
gcloud logs read "run.googleapis.com%2F$SERVICE_NAME"   --limit=50   --project="$PROJECT_ID"

# Ver revisiones del servicio
gcloud run revisions list   --service="$SERVICE_NAME"   --region="$REGION"
```

---

## 10. Limpieza (para evitar costos innecesarios)

Si quieres borrar todo lo creado para este servicio:

```bash
# 1) Eliminar el servicio de Cloud Run
gcloud run services delete "$SERVICE_NAME" --region="$REGION" --quiet

# 2) (Opcional) Eliminar el repositorio de imágenes (borra todas las imágenes dentro)
gcloud artifacts repositories delete "$REPO_NAME"   --location="$REGION"   --quiet
```

Si solo quieres borrar imágenes específicas, puedes hacerlo desde la consola web de Artifact Registry o usando:

```bash
gcloud artifacts docker images delete   "$IMAGE_URI@DIGEST_O_TAG_QUE_QUIERAS_BORRAR"
```

---

## 11. Notas de costos (resumen rápido)

- Cloud Run cobra por CPU, memoria y requests mientras el contenedor está recibiendo tráfico; tiene un free tier generoso.
- Cloud Build también tiene una capa gratuita de minutos de build.
- Artifact Registry cobra por almacenamiento de imágenes y egress, con un nivel gratuito pequeño.

Es buena idea:

- Revisar periódicamente la sección de costos de tu proyecto en la consola de GCP.
- Configurar alertas de presupuesto para evitar sorpresas.
