# DOCKER TUTORIAL
Guys, when you set up your gcloud instance, please try to use [docker](https://opensource.com/resources/what-docker), a quite lightweight virtual machine, which greatly simplifies the process of configuring environment.

# step 0.
Set up your gcloud instance with GPU properly, I recommend to use the [DeepLearning VM](https://console.cloud.google.com/marketplace/details/click-to-deploy-images/deeplearning).

# step 1.
Install docker according to your OS version, just follow the steps [here](https://docs.docker.com/install/linux/docker-ce/debian/).
in cmd type `docker` to see if installed successfully.

# step 2.
install nvidia docker runtime, just follow these instructions
```
# Add the package repositories
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update

# Install nvidia-docker2 and reload the Docker daemon configuration
sudo apt-get install -y nvidia-docker2
sudo pkill -SIGHUP dockerd

# Test nvidia-smi with the latest official CUDA image
docker run --runtime=nvidia --rm nvidia/cuda:9.0-base nvidia-smi
```

# step 3.
create a docker volume to persist the data
```
docker volume create mlp-vol
```
pull our docker image from docker repo and run our container
```
docker run --runtime=nvidia -it --rm -v mlp-vol:/code/mlp_proj/logs jay0518/mlp_proj
```
Here we run the container and map the volume `mlp-vol` to the path `/code/mlp_proj/logs` in container.

# check our experiment results
```
docker volume inspect mlp-vol
```
get the mountpoint
```
ls /var/lib/docker/volumes/mlp-vol/_data
```
