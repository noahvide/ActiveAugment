## Data
This paper used the following datasets:
- CIFAR-10
- CIFAR-100
- MNIST
- STL
- BRISC
- BUSI
- FETAL-PLANES
- ISIC
You can use the methods in dataProcessing/get_data.ipynb to load data for CIFAR-10, CIFAR-100, MNIST and STL. The rest can be found online. This file also contains methods for selecting a subset to use and split into training, validation and test set. For the paper use seed 0 and select 1000 samples from each dataset (780 samples from BUSI). 
