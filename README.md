An implementation of [Cascade R-CNN: Delving into High Quality Object Detection](https://arxiv.org/abs/1712.00726). I only trained and tested on pascal voc dataset. The source code is [here](https://github.com/zhaoweicai/cascade-rcnn) which implemented by caffe and also evalated on pascal voc.

## Introduction

As we all know,  the cascade structure is designed for R-CNN structure, so i just used the cascade structure based on [DetNet](https://arxiv.org/abs/1804.06215) to train and test on pascal voc dataset (DetNet is not only faster than fpn-resnet101, but also better than fpn-resnet101).

Based on [**DetNet_Pytorch**](https://github.com/guoruoqian/DetNet_pytorch), i mainly changed the forward function in fpn.py. It‘s just a naive implementation, so its speed is not fast. 

## Benchmarking

I benchmark this code thoroughly on pascal voc2007 and 07+12. Below are the results:

1). PASCAL VOC 2007 (Train/Test: 07trainval/07test, scale=600, ROI Align)

| model（FPN）     | GPUs            | Batch Size | lr   | lr_decay | max_epoch | Speed/epoch | Memory/GPU | AP   | AP50 | AP75 |
| ---------------- | --------------- | ---------- | ---- | -------- | --------- | ----------- | ---------- | ---- | ---- | ---- |
| DetNet59         | 1 GTX 1080 (Ti) | 2          | 1e-3 | 10       | 12        | 0.89hr      | 6137MB     | 44.8 | 76.1 | 46.2 |
| DetNet59-Cascade | 1 GTX 1080 (Ti) | 2          | 1e-3 | 10       | 12        | 1.62hr      | 6629MB     | 48.9 | 75.9 | 53.0 |

2). PASCAL VOC 07+12 (Train/Test: 07+12trainval/07test, scale=600, ROI Align)

| model（FPN）     | GPUs            | Batch Size | lr   | lr_decay | max_epoch | Speed/epoch | Memory/GPU | AP   | AP50 | AP75 |
| ---------------- | --------------- | ---------- | ---- | -------- | --------- | ----------- | ---------- | ---- | ---- | ---- |
| DetNet59         | 1 GTX 1080 (Ti) | 1          | 1e-3 | 10       | 12        | 2.41hr      | 9511MB     | 53.0 | 80.7 | 58.2 |
| DetNet59-Cascade | 1 GTX 1080 (Ti) | 1          | 1e-3 | 10       | 12        | 4.60hr      | 1073MB     | 55.6 | 80.1 | 61.0 |

## Preparation

First of all, clone the code

```
git clone https://github.com/guoruoqian/cascade-rcnn_Pytorch.git
```

Then, create a folder:

```shell
cd cascade-rcnn_Pytorch && mkdir data
```

### prerequisites

- Python 2.7 or 3.6
- Pytorch 0.2.0 or higher（not support pytorch version >=0.4.0）
- CUDA 8.0 or higher
- tensorboardX

### Data Preparation

- VOC2007: Please follow the instructions in [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to prepare VOC datasets. Actually, you can refer to any others. After downloading the data, creat softlinks in the folder data/.
- VOC 07 + 12: Please follow the instructions in [YuwenXiong/py-R-FCN](https://github.com/YuwenXiong/py-R-FCN/blob/master/README.md#preparation-for-training--testing) . **I think this instruction is more helpful to prepare VOC datasets.**

### Pretrained Model 

 You can download the detnet59 model which i trained on ImageNet from:

- detnet59: [dropbox](https://www.dropbox.com/home/DetNet?preview=detnet59.pth)，[baiduyun](https://pan.baidu.com/s/14_ztsAKcrZGb4nnm8aCMyQ)

 Download it and put it into the data/pretrained_model/. 

### Compilation

As pointed out by [ruotianluo/pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn), choose the right `-arch` in `make.sh` file, to compile the cuda code: 

| GPU model                  | Architecture |
| :------------------------- | :----------: |
| TitanX (Maxwell/Pascal)    |    sm_52     |
| GTX 960M                   |    sm_50     |
| GTX 1080 (Ti)              |    sm_61     |
| Grid K520 (AWS g2.2xlarge) |    sm_30     |
| Tesla K80 (AWS p2.xlarge)  |    sm_37     |

Install all the python dependencies using pip: 

```shell
pip install -r requirements.txt
```

Compile the cuda dependencies using following simple commands: 

```shell
cd lib
sh make.sh
```

It will compile all the modules you need, including NMS, ROI_Pooing, ROI_Align and ROI_Crop. The default version is compiled with Python 2.7, please compile by yourself if you are using a different python version. 

## Usage

If you want to use cascade structure, you must set  `--cascade`  and  `--cag` in the below script. `cag` determine whether perform class_agnostic bbox regression. 

train voc2007 use cascade structure:

```shell
CUDA_VISIBLE_DEVICES=3 python3 trainval_net.py exp_name --dataset pascal_voc --net detnet59 --bs 2 --nw 4 --lr 1e-3 --epochs 12 --save_dir weights --cuda --use_tfboard True --cag --cascade
```

test voc2007:

```shell
CUDA_VISIBLE_DEVICES=3 python3 test_net.py exp_name --dataset pascal_voc --net detnet59 --checksession 1 --checkepoch 7 --checkpoint 5010 --cuda --load_dir weights --cag --cascade
```

Before training voc07+12, you must set ASPECT_CROPPING in detnet59.yml False, or you will encounter some error during the training. 

train voc07+12:

```shell
CUDA_VISIBLE_DEVICES=3 python3 trainval_net.py exp_name2 --dataset pascal_voc_0712 --net detnet59 --bs 1 --nw 4 --lr 1e-3 --epochs 12 --save_dir weights --cuda --use_tfboard True --cag --cascade
```