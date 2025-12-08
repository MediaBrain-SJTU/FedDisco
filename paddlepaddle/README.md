# FedDisco: Federated Learning with Discrepancy-Aware Collaboration, ICML2023

This repo is the pytorch implementation of ICML2023 paper "FedDisco: Federated Learning with Discrepancy-Aware Collaboration". [PMLR Link](https://proceedings.mlr.press/v202/ye23f.html)

## Prerequisites
- Python 3.9

- CUDA 11.3

- Pytorch 1.10.2

- Torchvision 0.11.3

  Please refer to `requirements.txt` for more details. This code should work on most builds.

## Preparing dataset and model

The argument `--datadir` in `args.py` specifies the location of dataset. 

The argument `--model` has three valid values: `resnet18_gn` , `simple-cnn` and  `simple-cnn-mnist` . The first one is ResNet for HAM10000 dataset while other two are simple CNN networks for small-scale dataset with RGB or grayscale images.

## Federated training
We provide several shell scripts for training in several settings. The format is

```shell
sh disco_sh/$DATASET_$PARTITION.sh
```

For example, to train on Fashion-MNIST dataset with NIID-1 partition, run

```shell
sh disco_sh/fmnist_1.sh
```

 To train on CIFAR-10 dataset with NIID-2 partition, run

```shell
sh disco_sh/cifar10_2.sh
```

Note that  `--alg` specifies an algorithm.

## Tuning parameters of Disco

To run the experiments of baselines, please keep `--disco=0`. To integrate with our FedDisco, let `--disco=1` and change the value of `--disco_a` and `--disco_b`.

## Citation

Please cite our paper if you find the repository helpful. See other projects and papers at [Rui Ye's Homepage](https://rui-ye.github.io/).

```
@article{ye2023feddisco,
  title={FedDisco: Federated Learning with Discrepancy-Aware Collaboration},
  author={Ye, Rui and Xu, Mingkai and Wang, Jianyu and Xu, Chenxin and Chen, Siheng and Wang, Yanfeng},
  journal={arXiv preprint arXiv:2305.19229},
  year={2023}
}
```