Based on [jwyang/fpn.pytorch](https://github.com/jwyang/fpn.pytorch), i change little code to get a more reasonable mAP when training pascal voc 2007 and 07+12.
Pytorch implementation of Feature Pyramid Network (FPN) for Object Detection.



## Introduction

This project inherits the property of our [jwyang/fpn.pytorch](https://github.com/jwyang/fpn.pytorch).Hence, you can see more information about it.The following things are what I did :

* **The stride of Resnet layer4 change 2 from 1**. The most fundamental reason why mAP is low is that the anchor's position and number of each layer are calculated by stride in this code.The designed FPN_FEAT_STRIDES in config is [4, 8, 16, 32, 64].  When layer4's stride is set to 1, FPN_FEAT_STRIDES should be changed to [4, 8, 16, 16, 32], but FPN_FEAT_STRIDES is still the default value, which results in p5, p6 has about 3/4 of the anchors generated outside the image.
* **Changing loge to log2 in  _PyramidRoI_Feat**.In original paper, roi pool on pyramid feature maps using log2. It does not seem to affect the training results.
* **It supports training VOC07+12**.In the original code, in order to batch training and memory efficient, it crop the original image.When i train VOC07+12, i find some images don't have target object duo to the operation of crop. So i add a paramter ASPECT_CROPPING in config.py, set it False , it will not crop the images. So you can train VOC07 + 12.
* **It supports both python2 and python3.**

## Benchmarking

I benchmark this code thoroughly on pascal voc2007 and 07+12. Below are the results:

1). PASCAL VOC 2007 (Train/Test: 07trainval/07test, scale=600, ROI Align， 

model    | GPUs | Batch Size | lr        | lr_decay | max_epoch     |  Speed/epoch | Memory/GPU | mAP 
---------|-----------|----|-----------|-----|-----|-------|--------|--------
Res-101    | 1  GTX 1080 (Ti) | 2 | 1e-3 | 10  | 12  |  0.22 hr | 6137MB | 75.7 

2). PASCAL VOC 07+12 (Train/Test: 07+12trainval/07test, scale=600, ROI Align)



| model   | GPUs             | Batch Size | lr   | lr_decay | max_epoch | Speed/epoch | Memory/GPU | mAP  |
| ------- | ---------------- | ---------- | ---- | -------- | --------- | ----------- | ---------- | ---- |
| Res-101 | 1  GTX 1080 (Ti) | 1          | 1e-3 | 10       | 12        | \           | 9011MB     | 80.5 |

## Preparation

First of all, clone the code

```
git clone https://github.com/guoruoqian/FPN_Pytorch.git
```

Then, create a folder:

```
cd FPN_Pytorch && mkdir data
```

### prerequisites

- Python 2.7 or 3.6
- Pytorch 0.2.0 or higher
- CUDA 8.0 or higher
- tensorboardX

### Data Preparation

* VOC2007: Please follow the instructions in [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to prepare VOC datasets. Actually, you can refer to any others. After downloading the data, creat softlinks in the folder data/.
* VOC 07 + 12: Please follow the instructions in [YuwenXiong/py-R-FCN](https://github.com/YuwenXiong/py-R-FCN/blob/master/README.md#preparation-for-training--testing) . **I think this instruction is more helpful to prepare VOC datasets.**

### Pretrained Model & Compilation

​	Please follow the instructions in [Pretrained Model](https://github.com/jwyang/faster-rcnn.pytorch#pretrained-model) and [Compilation](https://github.com/jwyang/faster-rcnn.pytorch#compilation).

## Usage

train voc2007:

```
CUDA_VISIBLE_DEVICES=3 python3 trainval_net.py exp_name --dataset pascal_voc --net res101 --bs 2 --num_workers 4 --lr 1e-3 --epochs 12 --save_dir weights --cuda --use_tfboard True
```

test voc2007:

```
CUDA_VISIBLE_DEVICES=3 python3 test_net.py exp_name --dataset pascal_voc --net res101 --checksession 1 --checkepoch 7 --checkpoint 5010 --cuda --load_dir weights
```

train voc07+12:

```
CUDA_VISIBLE_DEVICES=3 python3 trainval_net.py exp_name2 --dataset pascal_voc_0712 --net res101 --bs 2 --num_workers 4 --lr 1e-3 --epochs 12 --save_dir weights --cuda --use_tfboard True
```

