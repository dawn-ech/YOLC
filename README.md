# You Only Look Clusters for Tiny Object Detection in Aerial Images

This is the implementation of "YOLC: You Only Look Clusters for Tiny Object Detection in Aerial Images".[[Paper](https://arxiv.org/abs/2404.06180)]

<p align="center">
    <img src="framework.jpg"/>
</p>

## Requirement
This repo is implemented based on [mmdetection](https://github.com/open-mmlab/mmdetection). Please refer to [get_started.md](https://github.com/open-mmlab/mmdetection/blob/v2.26.0/docs/en/get_started.md)

	- Python >= 3.8
	- PyTorch >= 1.7.0
	- mmdetection == 2.26.0 (>=2.17.0, <3.0.0)
	- kornia == 0.6.9

##  Train
#### Training on a single GPU
```
python train.py configs/yolc.py
```

#### Training on multiple GPUs
```
./dist_train.sh configs/yolc.py <your_gpu_num>
```

## Citation
If you find our paper is helpful, please consider citing our paper:
```BibTeX
@article{liu2024yolc,
  title={YOLC: You Only Look Clusters for Tiny Object Detection in Aerial Images},
  author={Liu, Chenguang and Gao, Guangshuai and Huang, Ziyue and Hu, Zhenghui and Liu, Qingjie and Wang, Yunhong},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2024},
  publisher={IEEE}
}
```