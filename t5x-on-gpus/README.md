# Running T5X workloads on GPUs

## Environment setup

TBD

- Create a Standard cluster. (There are issue with DNS in Autopilot. When resolved an Autopilot will be a perfect fit for this scenario)
- Create a GPU node pool. Two nodes. Each equipped with two T4 GPUs.


### Connect to the cluster

```
gcloud container clusters get-credentials $CLUSTER_NAME \
    --region $REGION \
    --project=$PROJECT_ID
```


## Running JAX Hello World


### Build JAX docker container

```
cd ~/ml-on-gke/t5x-on-gke

docker build -t gcr.io/jk-mlops-dev/jax-sandbox .
docker push gcr.io/jk-mlops-dev/jax-sandbox 
```

### Run a job

```
cd ~/ml-on-gke/t5x-on-gke/hello-world
```

Update `kustomization.yaml` with your image URI.

```
kubectl apply -k ./
```

To verify. Get pods

```
kubectl get pods
```

Display logs for the pods after they complete running. You should see something similar to:

```
(base) jarekk@jk-tx5-dev:~$ kubectl logs jax-hello-world-1-d4ld4
I0224 22:44:20.062100 140423841105728 distributed.py:79] Connecting to JAX distributed service on 10.108.4.20:1234
I0224 22:44:21.066458 140423841105728 xla_bridge.py:355] Unable to initialize backend 'tpu_driver': NOT_FOUND: Unable to find driver in registry given worker: 
I0224 22:44:21.579067 140423841105728 xla_bridge.py:355] Unable to initialize backend 'rocm': NOT_FOUND: Could not find registered platform with name: "rocm". Available platform names are: Interpreter Host CUDA
I0224 22:44:21.579529 140423841105728 xla_bridge.py:355] Unable to initialize backend 'tpu': module 'jaxlib.xla_extension' has no attribute 'get_tpu_client'
I0224 22:44:21.579663 140423841105728 xla_bridge.py:355] Unable to initialize backend 'plugin': xla_extension has no attributes named get_plugin_device_client. Compile TensorFlow with //tensorflow/compiler/xla/python:enable_plugin_device set to true (defaults to false) to enable this.
Coordinator host name: jax-hello-world-0.headless-svc
Coordiantor IP address: 10.108.4.20
JAX global devices:[StreamExecutorGpuDevice(id=0, process_index=0, slice_index=0), StreamExecutorGpuDevice(id=1, process_index=0, slice_index=0), StreamExecutorGpuDevice(id=2, process_index=1, slice_index=1), StreamExecutorGpuDevice(id=3, process_index=1, slice_index=1)]
JAX local devices:[StreamExecutorGpuDevice(id=2, process_index=1, slice_index=1), StreamExecutorGpuDevice(id=3, process_index=1, slice_index=1)]
[4. 4.]
Hooray ...
```