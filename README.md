# FedDisco: Federated Learning with Discrepancy-Aware Collaboration, ICML2023

This repo is the pytorch implementation of ICML2023 poster "FedDisco: Federated Learning with Discrepancy-Aware Collaboration".

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

