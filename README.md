# Strategic Data Navigation: Information Value-based Sample Selection
This repository contains the implementation code for our paper.

## Authors
- Csanád L. Balogh<sup>1,3</sup>
  - [Google Scholar](https://scholar.google.com/citations?user=p0eQRMAAAAAJ&hl=hu&oi=ao)
  - [ORCID](https://orcid.org/0009-0003-2481-8456)
- Bálint Pelenczei<sup>2</sup>
  - [Google Scholar](https://scholar.google.com/citations?user=4y2OD3QAAAAJ&hl=hu&oi=ao)
  - [ORCID](https://orcid.org/0000-0001-9194-8574)
  - [SZTAKI](https://sztaki.hun-ren.hu/en/balint-pelenczei)
- Bálint Kővári<sup>1,3</sup>
  - [Google Scholar](https://scholar.google.com/citations?user=WrtttXEAAAAJ&hl=hu&oi=ao)
  - [ORCID](https://orcid.org/0000-0003-2178-2921)
- Tamás Bécsi<sup>1</sup>
  - [Google Scholar](https://scholar.google.com/citations?user=Sdw_b5YAAAAJ&hl=hu&oi=ao)
  - [ORCID](https://orcid.org/0000-0002-1487-9672)
  - [Department](http://www.kjit.bme.hu/index.php/en/tanszeki-munkatarsak-en/12-munkatarsak/munkatarsak-angol/259-becsi-tamas)

<sup>1</sup>Department of Control for Transportation and Vehicle Systems, Faculty of Transportation Engineering and Vehicle Engineering, Budapest University of Technology and Economics, Budapest, 1111, Hungary

<sup>2</sup>Systems and Control Laboratory, HUN-REN Institute for Computer Science and Control (SZTAKI), Budapest, 1111, Hungary

<sup>3</sup>Asura Technologies Ltd., Budapest, 1122, Hungary

## Abstract
Artificial Intelligence represents a rapidly expanding domain, with several industrial applications demonstrating its superiority over traditional techniques. Despite numerous advancements within the subfield of Machine Learning, it encounters persistent challenges, highlighting the importance of ongoing research efforts. Among its primary branches, this study delves into two categories, being Supervised and Reinforcement Learning, particularly addressing the common issue of data selection for training. The inherent variability in informational content among data points is apparent, wherein certain samples offer more valuable information to the neural network than others. However, evaluating the significance of various data points remains a non-trivial task, generating the need for a robust method to effectively prioritize samples. Drawing inspiration from Reinforcement Learning principles, this paper introduces a novel sample prioritization approach, applied to Supervised Learning scenarios, aimed at enhancing classification accuracy through strategic data navigation, while exploring the boundary between Reinforcement and Supervised Learning techniques. We provide a comprehensive description of our methodology, while revealing the identification of an optimal prioritization balance and demonstrating its beneficial impact on model performance. Although classification accuracy serves as the primary validation metric, the concept of information density-based prioritization encompasses wider applicability. Additionally, the paper investigates parallels and distinctions between Reinforcement and Supervised Learning methods, declaring that the foundational principle is equally relevant, hence completely adaptable to Supervised Learning with appropriate adjustments due to different learning frameworks.

## Repo structure
### Folders
- `agent/` - contains agent-related components (lr schedule, loss)
- `callbacks/` - includes callback functions
- `data/` - stores data files or datasets as per described below
- `dataset/` - handling of dataset processing and management
- `factory/` - contains creator functions for different parsed configs
- `trainer/` - holds the main training loop module
- `training_functions/` - contains utility functions for training
### Files
- `Dockerfile` - defines Docker image configuration for containerization
- `configuration.ini` - stores configuration settings
- `docker-compose.yml` - defines Docker services configuration
- `requirements.txt` - lists project dependencies
- `train.py` - main training script
### Datasets
```
|- ... other folders ...
|   |- ... other files ...
|- data
|   |- link_to_download
|   |- download_tiny_imagenet.py
|   |- Cifar10
|   |   |- [...]
|   |- Cifar100
|   |   |- [...]
|   |- TinyImagenet
|   |   |- [...]
|   |- Imagenet
|   |   |- [...]
 ```

## Setup
In order to run any demonstration or training code included in this repository under the same circumstances, and to have all the dependencies installed, issue the following commands:

*Requirements*: [`docker`](https://docs.docker.com/get-docker/) or [`nvidia-docker`](https://github.com/NVIDIA/nvidia-docker), and [`docker compose`](https://github.com/docker/compose)  installed on your computer.

Clone this repository by : 

```bash
git clone https://github.com/csanadlb/sl_prioritized_sampling
cd sl_prioritized_sampling
```

To build the Docker image and run the container with predefined parameters and settings, run the following commands:

```bash
docker build . --tag sl_prioritization
docker compose up -d
```

In order to enter the running container through SSH, run:

```bash
docker exec -it sl_prioritization /bin/bash
```

If your machine has an Nvidia GPU and `nvidia-docker` is set up, replace `docker` with `nvidia-docker` in the previous commands.

### Data preparation
In order to gather the required public datasets, run the commands as follows:
>For **Cifar10**, run:
```bash
cd data && mkdir Cifar10 && cd Cifar10 && wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz && tar -xvf cifar-10-python.tar.gz && cd ../..
```
>For **Cifar100**, run:
```bash
cd data && mkdir Cifar100 && cd Cifar100 && wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz && cd ../..
```
>For **TinyImagenet**, run:
```bash
cd data && python3 download_tiny_imagenet.py && cd ..
```
>For **ImageNet**:
Download from official site: https://www.image-net.org/. Registration is required

## Training
In order to start training, take the following steps after entering the Docker container via SSH:
1. Open the config file with `nano` or an equivalent tool, and modify run parameters as desired:
```bash
nano configuration.ini
```
2. Run the main training script by:
```bash
python3 train.py
```

## Citation
Please use this bibtex if you would like to cite our work in your publications:

```bibtex
@article{,
  author={},
  title={},
  journal={},
  year={}
}
```