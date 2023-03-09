# Docker

## Create a Dockerfile

Open the `Dockerfile` to understand what it does.

## Build container locally

`CONTAINER_NAME` is the name we are giving our container in this example.

```bash
docker build -t CONTAINER_NAME .
```
### You can run container shell

```bash
docker run -it CONTAINER_NAME sh
```

### You can run container locally

```bash
docker run -p 8080:8000 CONTAINER_NAME
```

And connect to it: http://localhost:8080/

### List running containers

```bash
docker ps
```

### Stop container

```bash
docker stop <container_id>
```

or

```bash
docker kill <container_id>
```

## Push to Google Container Registry

Go to the GCP console, to Container Registry, and enable the service.

### Authenticate docker

```bash
gcloud auth configure-docker
```

### Build for deployment

You need to define your project ID.

```bash
# This section creates the necessary variables to build
export GCP_PROJECT_ID="PROJECT_ID"
export DOCKER_IMAGE_NAME="CONTAINER_NAME"
export GCR_MULTI_REGION="eu.gcr.io"
export GCR_REGION="europe-west1"

# This uses the variables to build
docker build -t $GCR_MULTI_REGION/$GCP_PROJECT_ID/$DOCKER_IMAGE_NAME .
```

⚠️ APPLE M1 Silicon users must build like this:

```bash
docker build --platform linux/amd64 -t $GCR_MULTI_REGION/$GCP_PROJECT_ID/$DOCKER_IMAGE_NAME .
```

At this point, check that it works locally before final push:

```bash
docker run -e PORT=8000 -p 8080:8000 $GCR_MULTI_REGION/$GCP_PROJECT_ID/$DOCKER_IMAGE_NAME
```

And if it does, you can finally push to Google Container Registry.

### PUSH!

```bash
# Don't foget to replace with your own PROJECT_ID and CONTAINER_NAME
export GCP_PROJECT_ID="PROJECT_ID"
export DOCKER_IMAGE_NAME="CONTAINER_NAME"
export GCR_MULTI_REGION="eu.gcr.io"
export GCR_REGION="europe-west1"

docker push $GCR_MULTI_REGION/$GCP_PROJECT_ID/$DOCKER_IMAGE_NAME
```

## Cloud run

Finally, you need to make your api continuously avilable. Go to the GCP console, to Cloud Run, and enable the service. You might also have to enable Cloud Run API.

```bash
export GCP_PROJECT_ID="PROJECT_ID"
export DOCKER_IMAGE_NAME="CONTAINER_NAME"
export GCR_MULTI_REGION="eu.gcr.io"
export GCR_REGION="europe-west1"

# Deploy your API to cloud run
gcloud run deploy --image $GCR_MULTI_REGION/$GCP_PROJECT_ID/$DOCKER_IMAGE_NAME --platform managed --region $GCR_REGION
```
