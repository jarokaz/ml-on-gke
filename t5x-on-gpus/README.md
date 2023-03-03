# Running T5X workloads on GPUs

## Environment setup

TBD

- Create a Standard cluster. (There are issue with DNS in Autopilot. When resolved Autopilot will be a perfect fit for this scenario)
- Create a GPU node pool. Two nodes. Each equipped with two T4 GPUs.
- Create a GCS bucket for jobs staging


### Connect to the cluster

```
gcloud container clusters get-credentials $CLUSTER_NAME \
    --region $REGION \
    --project=$PROJECT_ID
```


## Running a simple T5X job

### Install TFDS

```
pip install  tfds-nightly 
```

### Prepare the `wmt_t2t_translate` dataset

```
BUCKET_NAME=<YOUR_BUCKET_NAME>
export TFDS_DATA_DIR=gs://${BUCKET_NAME}/datasets
```

```
tfds build --data_dir $TFDS_DATA_DIR --experimental_latest_version wmt_t2t_translate
```

### Build T5X docker container

```
cd ~/ml-on-gke/t5x-on-gke

docker build -t <YOUR IMAGE URI> .
docker push <YOUR IMAGE URI> 
```


### Run a job

```
cd ~/ml-on-gke/t5x-on-gke/

```

Update `kustomization.yaml` 
- with your image URI.
- your job name


Update `job-path.yaml` with your T5X job parameters. E.g. :
- run_mode
- tfds_data_dir
- gin_file
- model_dir
- etc.


```
kubectl apply -k ./
```

To verify. Get pods

```
kubectl get pods
```


To delete the job

```
kubectl delete -k ./
```
