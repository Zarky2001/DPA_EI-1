# DPA_EI
Yan Zhao, Wenwei He, and Hong Zhao*  DPA-EI: Long-tailed classification by dual progressive augmentation from explicit and implicit perspectives
## Prerequisite

- PyTorch >= 1.2.0
- Python3
- torchvision
- PIL
- argparse
- numpy

## Evaluation

We provide several trained models of MetaSAug for evaluation.

Testing on CIFAR-LT-10/100:

- `sh scripts/MetaSAug_CE_test.sh`
- `sh scripts/MetaSAug_LDAM_test.sh`

Testing on ImageNet and iNaturalist18:

- `sh ImageNet_iNat/test.sh`

The trained models are in [Google Drive](https://drive.google.com/drive/folders/1YyE4RAniebDo8KyvdobcRfS0w5ZtMAQt?usp=sharing).

## Getting Started

### Dataset
- Long-tailed CIFAR10/100: The long-tailed version of CIFAR10/100. Code for coverting to long-tailed version is in [data_utils.py](https://github.com/BIT-DA/MetaSAug/blob/main/data_utils.py).
- ImageNet-LT: The long-tailed version of ImageNet. [[Long-tailed annotations](https://github.com/BIT-DA/MetaSAug/tree/main/ImageNet_iNat/data)]
- [iNaturalist2017](https://github.com/visipedia/inat_comp/tree/master/2017): A natural long-tailed dataset.
- [iNaturalist2018](https://github.com/visipedia/inat_comp/tree/master/2018): A natural long-tailed dataset.

### Training

Training on CIFAR-LT-10/100:
```
CIFAR-LT-100, MetaSAug with LDAM loss
python3.6 MetaSAug_LDAM_train.py --gpu 0 --lr 0.1 --lam 0.75 --imb_factor 0.05 --dataset cifar100 --num_classes 100 --save_name MetaSAug_cifar100_LDAM_imb0.05 --idx 1
```

Or run the script:

```
sh scripts/MetaSAug_LDAM_train.sh
```

Training on ImageNet-LT:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 53212 train.py  --lr 0.0003 --meta_lr 0.1 --workers 0 --batch_size 256 --epochs 20 --dataset ImageNet_LT --num_classes 1000 --data_root ../ImageNet
```

Or run the script:

```
sh ImageNet_iNat/scripts/train.sh
```

**Note**: Training on large scale datasets like ImageNet-LT and iNaturalist2017/2018 involves multiple gpus for faster speed. To achieve better generalizable representations, vanilla CE loss is used for training the network in the early training stage. For convenience, the training starts from the pre-trained models, e.g., [ImageNet-LT](https://dl.fbaipublicfiles.com/classifier-balancing/ImageNet_LT/models/resnet50_uniform_e90.pth), [iNat18](https://dl.fbaipublicfiles.com/classifier-balancing/iNaturalist18/models/resnet50_uniform_e200.pth) (both from project [cRT](https://github.com/facebookresearch/classifier-balancing)).
