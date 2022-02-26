following the guide (gcp tpu vm pytorch xla)[https://cloud.google.com/tpu/docs/pytorch-xla-ug-tpu-vm]

install (gcloud cli)[https://cloud.google.com/sdk/docs/install]


provision a persistent disk
```
gcloud compute disks create --size=200GB --zone=$ZONE $PD_NAME --project=$PROJECT_ID
```

create a preprocessing worker
```
gcloud compute instances create pd-filler \
--zone=$ZONE \
--machine-type=n1-standard-16  \
--image-family=torch-xla \
--image-project=ml-images  \
--boot-disk-size=200GB \
--scopes=https://www.googleapis.com/auth/cloud-platform \
--disk=name=$PD_NAME,auto-delete=no
gcloud compute ssh pd-filler --zone=$ZONE
```

create a tpu vm
```
gcloud alpha compute tpus tpu-vm create tpu-name \
  --zone=zone \
  --accelerator-type=v3-8 \
  --version=v2-alpha
```

attach disk:
```
gcloud compute instances attach-disk $VM_NAME --disk $PD_NAME --zone $ZONE --mode=rw
```

ssh into the tpu vm
```
gcloud alpha compute tpus tpu-vm ssh tpu-name --zone zone --project project-id

```

show tpu vms:
```
gcloud alpha compute tpus tpu-vm list --zone=zone
```

show a tpu vm:
```
gcloud alpha compute tpus tpu-vm describe tpu-name \
  --zone=zone
```



install additional dependencies:
```
pip3 install -r /usr/share/tpu/models/official/requirements.txt

pip3 install --upgrade "cloud-tpu-profiler>=2.3.0"
pip3 install --user --upgrade -U "tensorboard>=2.3"
pip3 install --user --upgrade -U "tensorflow>=2.3"
```


```
export XRT_TPU_CONFIG="localservice;0;localhost:51011"
git clone secora
```
start xrt in separate process, to keep the compilation caches:
```
python3 -m torch_xla.core.xrt_run_server --port 51011 --restart

```

```
export PROJECT_ID=secora
export TPU_NAME=tpu-name
export ZONE=zone
export RUNTIME_VERSION=tpu-vm-pt-1.10
```

gcloud alpha compute tpus tpu-vm delete tpu-name \
--zone=zone


