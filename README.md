# FedDisco: Federated Learning with Discrepancy-Aware Collaboration, ICML2023

**Note**: This repo is still in progress.

## Introduction
Here we provide our codes of FedAvg on NIID-1 and NIID-2 settings of CIFAR-10. 

## Instructions
Please run the code of NIID-1 via:

> sh fedavg_10_1.sh

Please run the code of NIID-2 via:

> sh fedavg_10_2.sh

 Note that in `fedavg_10_1.sh` and `fedavg_10_2.sh`,  `--partition=noniid` equals to NIID-1 and `--partition=noniid-4` equals to NIID-2.

## Tuning Hyper-parameters

You may select your cuda device by changing the value of `CUDA_VISIBLE_DEVICES`. To run the experiments of baseline, please keep `--distribution_aware='not'`. To integrate with our FedDisco, let `--distribution_aware='sub'` and change the value of `--disco_a` and `--disco_b`, which corresponds to a and b in our paper, respectively. 



