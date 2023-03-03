# Running T5X workloads on GPUs

## Environment setup

### Create a service account for GKE Autopilot

```
export PROJECT_ID=jk-mlops-dev
export SA_NAME=autopilot-sa

gcloud iam service-accounts create $SA_NAME \
    --description="A Service Account for Autopilote GKE" \
    --display-name="Autopilot SA"
```

```
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/editor"
```


### Provision a GKE Autopilot cluster

```
export CLUSTER_NAME=jk-autopilot-1
export REGION=us-central1
export PROJECT_ID=jk-mlops-dev
export SUBNET_NAME=jk-autopilot-subnet
export NETWORK=default


gcloud container clusters create-auto $CLUSTER_NAME \
    --region $REGION \
    --project=$PROJECT_ID \
    --create-subnetwork name=$SUBNET_NAME \
    --network=$NETWORK \
    --service-account="$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com" \
    --scopes=cloud-platform

```

### Validate cluster

```
gcloud container clusters describe $CLUSTER_NAME \
    --region $REGION
```


### Connect to the cluster

```
gcloud container clusters get-credentials $CLUSTER_NAME \
    --region $REGION \
    --project=$PROJECT_ID
```

## Experiments

### Build JAX docker container

```
docker build -t gcr.io/jk-mlops-dev/jax-sandbox .
docker push gcr.io/jk-mlops-dev/jax-sandbox 
```