# covid-19-project

The actual medical data is required for initiating training. For accessing, please contact: kunlunhe@plagh.org


# a denseNet structure
A Python 3/TensorFlow implementation of denseNet (with few modifications) ([paper](https://arxiv.org/abs/1608.06993)). 

The deep models in this implementation are built on [TF-slim]

## Contents
1. [Requirements: Software](#requirements-software)
2. [Repo Organization](#repo-organization) 


## Requirements: Software
1. Ubuntu 16 
2. Python 3.6+: 
3. TensorFlow v1.14+: See [TensorFlow Installation with Anaconda](https://www.tensorflow.org/install/install_linux#InstallingAnaconda).
4. Some additional python packages you may or may not already have: `sklearn`, `easydict`, `matplotlib` `scipy`, `Pillow`, `tqdm`. These should all be pip installable before training (pip install [package]):


## Repo Organization
- assests: model related contents
- allReduceProtocal: multi-GPU training protocal
- helper: for data loading

