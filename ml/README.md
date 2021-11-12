# skytruth-internal


## fastai2_training setup (reproducing the current deployment)
Download this folder to local (it's 8Gb and contains the training and testing data): https://drive.google.com/drive/folders/1ih3gPWl_WquQ_QexbeG-GankPkVEGMmA

Conda is a standard method for setting up fastai2. after installing Miniconda for your system, install Mamba for fast solving and package downloads. Then:


If you have a gpu, use `nvcc -V` to check the cudatoolkit version you need to install and edit the `cudatoolkit` version in the code below. This was tested with CUDA 11.3, which matches with the latest stable release of pytorch, 1.10.
also make sure to install `nb_conda_kernels` in the base env where you start jupyter lab so that this conda env is discoverable.

for gpu, first do this to se up cuda
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda-repo-ubuntu2004-11-3-local_11.3.0-465.19.01-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-3-local_11.3.0-465.19.01-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2004-11-3-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
```

Then, set up the conda env for the gpu
```bash
conda create -n fastai2 -y
conda activate fastai2
mamba install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
mamba install -c fastchan fastai -y
mamba install -c conda-forge ipykernel ipywidgets -y
conda deactivate
conda install -n base -c conda-forge jupyterlab_widgets -y
jupyter lab
```

instructions for a cpu env are below. some pytorch errors come with more informative tracebacks on CPU.
```bash
conda create -n fastai2-cpu -y
conda activate fastai2-cpu
mamba install pytorch torchvision torchaudio cpuonly -c pytorch -y
mamba install -c fastchan fastai -y
mamba install -c conda-forge ipykernel ipywidgets -y
conda deactivate
conda install -n base -c conda-forge jupyterlab_widgets -y
jupyter lab
```

## icevision environment setup

Set up the conda env for the gpu
```bash
conda create -n icevision python=3.8 -y
conda activate icevision
git clone https://github.com/airctic/icevision.git
cd icevision
bash icevision_install.sh cuda11 master
mamba install -c conda-forge ipykernel ipywidgets -y
```

if you are on colab, see https://github.com/airctic/icevision/blob/master/install_colab.sh and the fastai2_training.ipynb in the drive folder for loading the data into the icevision notebook from google drive.