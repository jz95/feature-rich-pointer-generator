# DOCKER TUTORIAL
[docker](https://opensource.com/resources/what-docker) is a quite lightweight virtual machine, which greatly simplifies the process of configuring environment.

# step 1.
Install docker according to your OS version, just follow the steps [here](https://docs.docker.com/install/linux/docker-ce/debian/).
in cmd type `docker` to see if installed successfully.

# step 2.
Install nvidia docker runtime, just follow these instructions. 
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
Create a docker volume to persist the data
```
# create volume for experiment log
docker volume create my-vol-log
# create volume for data
docker volume create my-vol-data
```
Build an image from file named as frpg, and run the container
```
docker build -f dockerfile.gpu -t frpg .
docker run --runtime=nvidia -it --rm -v my-vol-log:/code/frpg/logs -v my-vol-data:/data frpg
```
Here we run the container and map the volume `my-vol-log` to the path `/code/frpg/logs` in container, and map `my-vol-data` to `/data` in container.

# check our experiment results
```
docker volume inspect my-vol-log
docker volume inspect my-vol-data
```
get the mountpoint
```
ls /var/lib/docker/volumes/my-vol-log/_data
```
