# YOLC

This is the implementation of "YOLC: You Only Look Clusters for Tiny Object Detection in Aerial Images".[[Paper](https://arxiv.org/abs/2404.06180)]



##
 The overal framework architecture
 -----------------------------------------------
 ![](./framework.png)

## Requirement
	- Python >= 3.8
	- PyTorch >= 1.7.0
	- mmdetection == 2.26.0 (>=2.17.0, <3.0.0)

##  Train

```
./dist_train.sh configs/yolc.py <your_gpu_num>
```