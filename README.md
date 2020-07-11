# CVND---Image-Captioning-Project

## Overview

This project from Udacity's nano-degree course "Computer Vision" is about setting up an Image Captioning system using Pytorch. The image captioning system consists of a Convolutional Neural Networks (CNNs) as encoder and an Recurrent Neural Network (RNN) based on LSTM cells as decoder. The CNN encoder network receives a color image and detects objects and feature, which are encoded and forwarded to the decoder LSTM network, which predicts image captions for the endoded objects. 

The image captioning system is trained on a part of Microsoft **C**ommon **O**bjects in **CO**ntext (MS COCO) dataset, which is a large-scale dataset for scene understanding. The dataset is commonly used to train and benchmark object detection, segmentation, and captioning algorithms.  

![Sample Dog Output](images/coco-examples.jpg)

You can read more about the dataset on the [website](http://cocodataset.org/#home) or in the [research paper](https://arxiv.org/pdf/1405.0312.pdf).

The project is structured as a series of Jupyter notebooks that are designed to be completed in sequential order (`0_Dataset.ipynb, 1_Preliminaries.ipynb, 2_Training.ipynb, 3_Inference.ipynb`).

__Notebook 0__ : Initialization of the COCO dataset [0_Dataset](0_Dataset.ipynb)  

__Notebook 1__ : Data pre-processing and design of an initial CNN-RNN network [1_Preliminaries](1_Preliminaries.ipynb)  

__Notebook 2__ : Training of the CNN-RNN network with different model structures and hyperparameters [2_Training_1](2_Training_1.ipynb) or [2_Training_2](2_Training_2.ipynb)  

__Notebook 3__ : Validation of the CNN-RNN network using greedy search and beam search in comparision [2_Validation_2](2_Validation_2.ipynb)

__Notebook 4__ : Test the trained CNN-RNN image captioning network on examples [3_Inference_1](3_Inference_1.ipynb) or [3_Inference_2](3_Inference_2.ipynb)


## Installation

### 1. Install Anaconda with Python

In order to run the notebook please download and install [Anaconda](https://docs.anaconda.com/anaconda/install/) with Python 3.6 on your machine. Further packages that are required to run the notebook are installed in a virtual environment using conda.


### 2. Create a Virtual Environment

In order to set up the prerequisites to run the project notebook you should create a virtual environment, e. g. using conda, Anaconda's package manager, and the following command

```
conda create -n computer-vision python=3.6
```

The virtual environment needs to be activated by

```
activate computer-vision
```

### 3. Download the project from github

You can download the project from github as a zip file to your Downloads folder from where you can unpack and move the files to your local project folder. Or you can clone from Github using the terminal window. Therefore, you need to prior install git on your local machine e. g. using

```
conda install -c anaconda git
```

When git is installed you can create your local project version. Clone the repository, and navigate to the download folder. This may take a minute or two to clone due to the included image data.

```
git clone https://github.com/AndiA76/image-captioning.git
cd image-captioning
```

### 4. Download COCO dataset and COCO API

1. Clone this repo: https://github.com/cocodataset/cocoapi  
```
git clone https://github.com/cocodataset/cocoapi.git  
```

2. Setup the coco API (also described in the readme [here](https://github.com/cocodataset/cocoapi)) 
```
cd cocoapi/PythonAPI  
make  
cd ..
```

3. Download some specific data from here: http://cocodataset.org/#download (described below)

* Under **Annotations**, download:
  * **2014 Train/Val annotations [241MB]** (extract captions_train2014.json and captions_val2014.json, and place at locations cocoapi/annotations/captions_train2014.json and cocoapi/annotations/captions_val2014.json, respectively)  
  * **2014 Testing Image info [1MB]** (extract image_info_test2014.json and place at location cocoapi/annotations/image_info_test2014.json)

* Under **Images**, download:
  * **2014 Train images [83K/13GB]** (extract the train2014 folder and place at location cocoapi/images/train2014/)
  * **2014 Val images [41K/6GB]** (extract the val2014 folder and place at location cocoapi/images/val2014/)
  * **2014 Test images [41K/6GB]** (extract the test2014 folder and place at location cocoapi/images/test2014/)
  
* Save the data in cocoapi or another directory on your system. 
  * In the dataloader script [dataloader_train_val_test.py](dataloader_train_val_test.py) you need to set the default value of the input argument `cocoapi_loc` to point to the path where the coco dataset is stored.
  
**Please change the path to the PythonAPI in `cocoapi/PythonAPI` to your local setting in the first code cell in all notebooks.**


### 5. Install Pytorch with GPU support

In order to run the project notebook you need to install Pytorch. If you wish to install Pytorch with GPU support you also need to take care of your CUDA version and some [dependencies with Pytorch](https://pytorch.org/get-started/previous-versions/). I have used Ubuntu 18.04 LTS with CUDA 10.0 and Python 3.6 to run the project notebook. Therefore you need to enter the following installation command:

CUDA 10.0
```
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
```

### 6. Further requirements 

Besides Pytorch you need to install a couple of further packages, which are required by the project notebook. The packages are specified in the [requirements.txt](requirements.txt) file (incl. OpenCV for Python). You can install them using pip resp. pip3:

```
pip install -r requirements.txt
```

### 7. Run the notebook

Now start a Jupyter notebook to run the project using following command

```
jupyter notebook
```

Navigate to your local project folder in the Jupyter notebook, open the notebooks 1...4

[0_Dataset](0_Dataset.ipynb)
[1_Preliminaries](1_Preliminaries.ipynb)
[2_Training_1](2_Training_1.ipynb) or [2_Training_2](2_Training_2.ipynb)
[2_Validation_2](2_Validation_2.ipynb)
[3_Inference_1](3_Inference_1.ipynb) or [3_Inference_2](3_Inference_2.ipynb)

and run them one after another.

