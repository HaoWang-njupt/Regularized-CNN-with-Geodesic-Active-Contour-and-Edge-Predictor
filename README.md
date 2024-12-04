


## Regularized CNN with Geodesic Active Contour and Edge Predictor for Image
Segmentation

Zhengmeng Jin, Hao Wang, Michael K.Ng, Lihua Min



[[Paper]()]


If our project is helpful for your research, please consider citing : 
``` 
@
        }
```


## Table of Content
* [1. Installation](#1-installation)
  * [1.1 Dependencies](#11-dependencies)
  * [1.2 Data](#12-data)
* [2. Quick Start](#2-quick-start)
  * [Training on the different datasets](#Training-on-the-different-datasets)
* [3. Testing](#3-testing)
  * [Training on the different datasets](#Synthetic_low_contrast)
  * [Other datasets](#Other-datasets) 
* [4. Acknowledgement](#4-acknowledgement)


## 1. Installation
### 1.1 Dependencies

This code was implemented with Python 3.10, PyTorch 2.1 and CUDA 12.3. Please refer to [the official installation](https://pytorch.org/get-started/previous-versions/). 





### 1.2 Data

We provide quick download commands in [DOWNLOAD_DATA.md](./DOWNLOAD_DATA.md) for Synthetic images, MRI images of Left Atrial and CT images of Liver.


## 2. Quick Start

### Training on the different datasets


```
python train_GAC.py --dataset Synthetic_low_contrast --img_size 256 --max_epochs 60 
python train_GAC.py --dataset Synthetic_SD0.1 --img_size 256 --max_epochs 60 
python train_GAC.py --dataset Synthetic_SD0.2 --img_size 256 --max_epochs 60 
python train_GAC.py --dataset Synthetic_SD0.3 --img_size 256 --max_epochs 60 
python train_GAC.py --dataset LA_MRI --img_size 224 --max_epochs 20
python train_GAC.py --dataset Liver_CT --img_size 224 --max_epochs 20
```



## 3. Testing
Following are the  results of **Our model** presented in the paper.



### Synthetic_low_contrast


```
python tese_GAC.py --dataset Synthetic_low_contrast --img_size 256 --max_epochs 60
```
<p>
  <img width="24%" alt="image" title="Image" src="examples\Synthetic_low_contrast\19image.png">
<img width="24%" alt="ground truth" title="Ground Truth" src="examples\Synthetic_low_contrast\19ground.png">
  <img width="24%" alt="results" title="Pred_segmentation" src="examples\Synthetic_low_contrast\19pred.png">
<img width="24%" alt="boundary" title="Pred_boundary" src="examples\Synthetic_low_contrast\19boundary.png">
</p>

### Other datasets

```
python test_GAC.py --dataset Synthetic_SD0.1 --img_size 256 --max_epochs 60 
python test_GAC.py --dataset Synthetic_SD0.2 --img_size 256 --max_epochs 60 
python test_GAC.py --dataset Synthetic_SD0.3 --img_size 256 --max_epochs 60 
python test_GAC.py --dataset LA_MRI --img_size 224 --max_epochs 20
python test_GAC.py --dataset Liver_CT --img_size 224 --max_epochs 20
```

## 4. Acknowledgement
The authors would like to thank the organization team of [MICCAI 2018 Atrial Segmentation Challenge](https://www.cardiacatlas.org/atriaseg2018-challenge/) and [AbdomenCT-1K](https://abdomenct-1k-fully-supervised-learning.grand-challenge.org/)  for the publicly available dataset. 
The code is built on top of [UNet](https://github.com/milesial/Pytorch-UNet) and [SegWithDistMap](https://github.com/JunMa11/SegWithDistMap). We would like to sincerely thanks those authors for their great works. 


