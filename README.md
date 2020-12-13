# Graph Attention-Based Deep Neural Network for 3D Point Cloud Processing
![image](https://github.com/Userac123/cc/blob/master/doc/overview_network.jpg)  
## Introduction
We proposes a novel end-to-end deep learning network for the features of disorder and irregularity of 3D point cloud data. Feature extraction stage consists of graph attention convolution and graph attention pooling. Graph attention convolution calculates the rela-
tionship weight by considering the importance of different center points to the neighborhood points, which reflects the spatial distribution relationship in the neighborhood area. Graph attention pooling not only considers the feature information of the sampling point but also merges the spatial distribution information in the neighborhood of the sampling point into the feature of the sampling point. The experimental results show that our network has competitive performance.
## Installation
Install [Tensorflow](https://www.tensorflow.org/install/). The code is tested under TF1.2 GPU version and Python 2.7 (version 3 should also work) on Ubuntu 14.04. There are also some dependencies for a few Python libraries for data processing and visualizations like cv2, h5py etc. It's highly recommended that you have access to GPUs.

### Compile Customized TF Operators
The TF operators are included under tf_ops, you need to compile them (check tf_xxx_compile.sh under each ops subfolder) first. Update nvcc and python path if necessary. The code is tested under TF1.2.0. If you are using earlier version it's possible that you need to remove the -D_GLIBCXX_USE_CXX11_ABI=0 flag in g++ command in order to compile correctly. To compile the operators in TF version >=1.4, you need to modify the compile scripts slightly.

## Usage
We used the same data in this paper as [pointnet++](https://github.com/charlesq34/pointnet2). Please go to data folder, follow instructions there and download the data. You are recommended to put unziped modelnet and shapenet folders in data/, but they can of course go somewhere else.
### - Shape Classification
* Run the training script:  
``` python train.py ```
* Run the evaluation script after training finished:  
``` python evalutate.py ```
### - Part Segmentation
Follow ModelNet instructions, but apply those in folder "part_seg", and you should be able to run the experiments smoothly.
