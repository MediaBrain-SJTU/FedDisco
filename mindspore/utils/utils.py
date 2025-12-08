import os
import logging
import copy
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
import numpy as np
from sklearn.metrics import confusion_matrix
from utils.model import *
from utils.datasets import CIFAR10_truncated, CIFAR100_truncated, FashionMNIST_truncated, ImageFolder_CINIC10, ImageFolder_HAM10000

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def init_nets(n_parties, args, device='cpu'):
    nets = {net_i: None for net_i in range(n_parties)}
    if args.dataset in {'cifar10', 'fmnist', 'cinic10'}:
        n_classes = 10
    elif args.dataset == 'ham10000':
        n_classes = 7
    elif args.dataset == 'cifar100':
        n_classes = 100
    for net_i in range(n_parties):
        net = ModelFedCon_noheader(args.model, args.out_dim, n_classes, args.dataset)
        # MindSpore models don't need explicit .to(device) usually, context handles it.
        # But we can keep them as objects.
        nets[net_i] = net
    return nets

def load_cifar10_data(datadir, download):
    # Transforms in MindSpore are usually applied in the dataset pipeline,
    # but here we are loading raw data for partitioning.
    # We can pass a simple callable if needed, or None.
    # The datasets.py classes handle transforms in __getitem__.
    
    # We need to define transforms using MindSpore vision ops if we want to match behavior.
    # However, since we are just loading data for partitioning here (getting y_train), transforms might not matter yet.
    # But the function returns X_train, y_train, X_test, y_test.
    # X_train should be transformed?
    # In original code: transform = transforms.Compose([transforms.ToTensor()])
    # ToTensor converts PIL/numpy (H,W,C) [0,255] to (C,H,W) [0.0, 1.0].
    
    transform = transforms.Compose([vision.ToTensor()])
    
    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=download, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=download, transform=transform)
    
    # Accessing .data directly from our custom class
    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target
    return (X_train, y_train, X_test, y_test)

def load_cinic10_data(datadir):
    transform = transforms.Compose([vision.ToTensor()])
    xray_train_ds = ImageFolder_CINIC10(datadir+'/train/', transform=transform)
    xray_test_ds = ImageFolder_CINIC10(datadir+'/sampled test/', transform=transform)

    X_train, y_train = np.array([s[0] for s in xray_train_ds.samples]), np.array([int(s[1]) for s in xray_train_ds.samples])
    X_test, y_test = np.array([s[0] for s in xray_test_ds.samples]), np.array([int(s[1]) for s in xray_test_ds.samples])
    return (X_train, y_train, X_test, y_test)

def load_HAM10000_data(datadir):
    transform = transforms.Compose([vision.ToTensor()])
    xray_train_ds = ImageFolder_HAM10000(datadir+'/train_new/', transform=transform)
    xray_test_ds = ImageFolder_HAM10000(datadir+'/val_new/', transform=transform)

    X_train, y_train = np.array([s[0] for s in xray_train_ds.samples]), np.array([int(s[1]) for s in xray_train_ds.samples])
    X_test, y_test = np.array([s[0] for s in xray_test_ds.samples]), np.array([int(s[1]) for s in xray_test_ds.samples])
    return (X_train, y_train, X_test, y_test)

def load_cifar100_data(datadir, download):
    transform = transforms.Compose([vision.ToTensor()])

    cifar100_train_ds = CIFAR100_truncated(datadir, train=True, download=download, transform=transform)
    cifar100_test_ds = CIFAR100_truncated(datadir, train=False, download=download, transform=transform)

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.target
    return (X_train, y_train, X_test, y_test)

def load_fmnist_data(datadir, download):
    transform = transforms.Compose([vision.ToTensor()])

    fmnist_train_ds = FashionMNIST_truncated(datadir, train=True, download=download, transform=transform)
    fmnist_test_ds = FashionMNIST_truncated(datadir, train=False, download=download, transform=transform)

    X_train, y_train = fmnist_train_ds.data, fmnist_train_ds.target
    X_test, y_test = fmnist_test_ds.data, fmnist_test_ds.target
    return (X_train, y_train, X_test, y_test)

def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts_dict = {}
    net_cls_counts_npy = np.array([])
    num_classes = int(y_train.max()) + 1

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts_dict[net_i] = tmp
        tmp_npy = np.zeros(num_classes)
        for i in range(len(unq)):
            tmp_npy[unq[i]] = unq_cnt[i]
        net_cls_counts_npy = np.concatenate(
                        (net_cls_counts_npy, tmp_npy), axis=0)
    net_cls_counts_npy = np.reshape(net_cls_counts_npy, (-1,num_classes))

    data_list=[]
    for net_id, data in net_cls_counts_dict.items():
        n_total=0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    print('Data distribution over clients (each row for a client): ')
    print(net_cls_counts_npy.astype(int), '\n')
    return net_cls_counts_npy

def partition_split_test_data(args):
    dataset, datadir, test_global_imbalanced, download = args.dataset, args.datadir, args.test_imb, args.download_data
    if dataset == 'cifar10':
        _, _, _, y_test = load_cifar10_data(datadir, download)
        class_num = 10
    elif dataset == 'cifar100':
        _, _, _, y_test = load_cifar100_data(datadir, download)
        class_num = 100
    elif dataset == 'fmnist':
        _, _, _, y_test = load_fmnist_data(datadir, download)
        class_num = 10
    elif dataset == 'cinic10':
        _, _, _, y_test = load_cinic10_data(datadir)
        class_num = 10

    labels = y_test
    n_test = y_test.shape[0]
    net_dataidx_map = {0: np.array([]).astype(int)}
    idxs = np.arange(n_test).astype(int)
    num_per_class = n_test // class_num

    # Sort labels
    idxs_labels = np.vstack((idxs, labels)).astype(int)
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    if test_global_imbalanced != 0: # Testing set is imbalanced, following an exponential decay
        imb_test_dist = []
        for i in range(class_num):
            num_k = int(num_per_class * (test_global_imbalanced ** (-i/(class_num-1))))
            net_dataidx_map[0] = np.concatenate(
                    (net_dataidx_map[0], idxs[num_per_class*i : (num_per_class*i+num_k)]), axis=0)
            imb_test_dist.append(num_k)

    testdata_cls_counts = record_net_data_stats(y_test, net_dataidx_map)
    return (net_dataidx_map, testdata_cls_counts, imb_test_dist)

def partition_data(args):
    # This function remains largely unchanged as it operates on numpy arrays
    dataset, datadir, partition, n_parties, beta, n_niid_parties, global_imbalanced, download = args.dataset, args.datadir, \
        args.partition, args.n_parties, args.beta, args.n_niid_parties, args.train_global_imb, args.download_data
    if dataset == 'cifar10':
        _, y_train, _, _ = load_cifar10_data(datadir, download)
    elif dataset == 'cifar100':
        _, y_train, _, _ = load_cifar100_data(datadir, download)
    elif dataset == 'fmnist':
        _, y_train, _, _ = load_fmnist_data(datadir, download)
    elif dataset == 'cinic10':
        _, y_train, _, _ = load_cinic10_data(datadir)
    elif dataset == 'ham10000':
        _, y_train, _, _ = load_HAM10000_data(datadir)
    n_train = y_train.shape[0]

    if partition == "homo" or partition == "iid":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    elif partition == "noniid-1":
        min_size = 0
        min_require_size = 10
        K = 10
        if dataset == 'cifar100':
            K = 100
        elif dataset == 'ham10000':
            K = 7
        N = y_train.shape[0]
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                if global_imbalanced!=0:   # Global dataset is imbalanced, following an exponential decay
                    ratio =  global_imbalanced
                    num_k = int(n_train/n_parties * (ratio ** (-k/(K-1))))
                    idx_k = idx_k[:num_k]
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif partition=='noniid-2' and dataset in ['cifar10', 'fmnist', 'cinic10']:
        labels = y_train
        num_non_iid_client = n_niid_parties
        num_iid_client =  n_parties-num_non_iid_client
        num_classes = int(labels.max()+1)
        num_sample_per_class = n_train//num_classes
        num_per_shard = int(n_train/num_classes/(num_non_iid_client+num_iid_client))
        net_dataidx_map = {i: np.array([]).astype(int) for i in range(n_parties)}
        idxs = np.arange(n_train).astype(int)

        # Sort labels
        idxs_labels = np.vstack((idxs, labels)).astype(int)
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]

        # Partition of non-iid clients
        for i in range(num_non_iid_client):
            net_dataidx_map[i] = np.concatenate(
                    (net_dataidx_map[i], idxs[((2*i)%10)*num_sample_per_class+num_per_shard*(i//5)*5:((2*i)%10)*num_sample_per_class+num_per_shard*(i//5+1)*5]), axis=0)
            net_dataidx_map[i] = np.concatenate(
                    (net_dataidx_map[i], idxs[((2*i+1)%10)*num_sample_per_class+num_per_shard*(i//5)*5:((2*i+1)%10)*num_sample_per_class+num_per_shard*(i//5+1)*5]), axis=0)
            np.random.shuffle(net_dataidx_map[i])
            net_dataidx_map[i] = list(net_dataidx_map[i])

        # Partition of iid clients
        for i in range(num_non_iid_client,n_parties):
            for j in range(num_classes):
                net_dataidx_map[i] = np.concatenate(
                        (net_dataidx_map[i], idxs[j*num_sample_per_class+num_per_shard*5*(num_non_iid_client//5)+num_per_shard*(i-num_non_iid_client): \
                                                    j*num_sample_per_class+num_per_shard*5*(num_non_iid_client//5)+num_per_shard*(i-num_non_iid_client+1)]), axis=0)
            np.random.shuffle(net_dataidx_map[i])
            net_dataidx_map[i] = list(net_dataidx_map[i])

    elif partition=='noniid-2' and dataset=='cifar100':
        labels = y_train
        num_non_iid_client = n_niid_parties
        num_iid_client =  n_parties-num_non_iid_client
        num_classes = int(labels.max()+1)
        num_sample_per_class = n_train//num_classes
        num_per_shard = int(n_train/num_classes/(num_non_iid_client+num_iid_client))

        net_dataidx_map = {i: np.array([]).astype(int) for i in range(n_parties)}
        idxs = np.arange(n_train).astype(int)

        # Sort labels
        idxs_labels = np.vstack((idxs, labels)).astype(int)
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]

        # Partition of non-iid clients
        for i in range(num_non_iid_client):
            for j in range(20):
                net_dataidx_map[i] = np.concatenate(
                        (net_dataidx_map[i], idxs[((20*i+j)%100)*num_sample_per_class+num_per_shard*(i//5)*5:((20*i+j)%100)*num_sample_per_class+num_per_shard*(i//5+1)*5]), axis=0)
                # print(((20*i+j)%100)*num_sample_per_class+num_per_shard*(i//5)*5,((20*i+j)%100)*num_sample_per_class+num_per_shard*(i//5+1)*5)
            np.random.shuffle(net_dataidx_map[i])
            net_dataidx_map[i] = list(net_dataidx_map[i])

        # Partition of iid clients
        for i in range(num_non_iid_client,n_parties):
            for j in range(num_classes):
                net_dataidx_map[i] = np.concatenate(
                        (net_dataidx_map[i], idxs[j*num_sample_per_class+num_per_shard*5*(num_non_iid_client//5)+num_per_shard*(i-num_non_iid_client): \
                                                    j*num_sample_per_class+num_per_shard*5*(num_non_iid_client//5)+num_per_shard*(i-num_non_iid_client+1)]), axis=0)
                # print(j*num_sample_per_class+num_per_shard*5*(num_non_iid_client//5)+num_per_shard*(i-num_non_iid_client),j*num_sample_per_class+num_per_shard*5*(num_non_iid_client//5)+num_per_shard*(i-num_non_iid_client+1))
            np.random.shuffle(net_dataidx_map[i])
            net_dataidx_map[i] = list(net_dataidx_map[i])

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    return (net_dataidx_map, traindata_cls_counts)

def set_client_from_params(mdl, params):
    # MindSpore parameters update
    # params is numpy array
    idx = 0
    for param in mdl.trainable_params():
        length = np.prod(param.shape)
        new_val = params[idx:idx+length].reshape(param.shape)
        param.set_data(mindspore.Tensor(new_val, param.dtype))
        idx += length
    return mdl

def get_mdl_params(model_list, n_par=None):
    if n_par==None:
        exp_mdl = model_list[0]
        n_par = 0
        for param in exp_mdl.trainable_params():
            n_par += np.prod(param.shape)

    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, mdl in enumerate(model_list):
        idx = 0
        for param in mdl.trainable_params():
            temp = param.data.asnumpy().reshape(-1)
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)

def compute_accuracy(model, dataloader, get_confusion_matrix=False, device="cpu", multiloader=False):
    # MindSpore evaluation
    model.set_train(False)
    
    correct, total = 0, 0
    true_labels_list, pred_labels_list = np.array([]), np.array([])
    
    # Loss function
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    
    loss_collector = []
    
    # Helper to process one batch
    def process_batch(x, target):
        x = mindspore.Tensor(x)
        target = mindspore.Tensor(target, mindspore.int32)
        
        out = model(x)
        # In MindSpore, SoftmaxCrossEntropyWithLogits expects (logits, labels)
        # If sparse=True, labels are indices.
        
        # out is (N, C)
        loss = criterion(out, target)
        
        # Pred
        # out.asnumpy()
        pred_label = np.argmax(out.asnumpy(), axis=1)
        target_np = target.asnumpy()
        
        return loss.asnumpy(), pred_label, target_np, x.shape[0]

    if multiloader:
        for loader in dataloader:
            # loader is a GeneratorDataset usually
            for data in loader.create_dict_iterator(output_numpy=True):
                x = data['data']
                target = data['label']
                
                loss_val, pred_label, target_np, batch_size = process_batch(x, target)
                
                loss_collector.append(loss_val)
                total += batch_size
                correct += (pred_label == target_np).sum()
                
                pred_labels_list = np.append(pred_labels_list, pred_label)
                true_labels_list = np.append(true_labels_list, target_np)
        avg_loss = sum(loss_collector) / len(loss_collector)
    else:
        for data in dataloader.create_dict_iterator(output_numpy=True):
            x = data['data']
            target = data['label']
            
            loss_val, pred_label, target_np, batch_size = process_batch(x, target)
            
            loss_collector.append(loss_val)
            total += batch_size
            correct += (pred_label == target_np).sum()
            
            pred_labels_list = np.append(pred_labels_list, pred_label)
            true_labels_list = np.append(true_labels_list, target_np)
        avg_loss = sum(loss_collector) / len(loss_collector)

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)
    
    # model.set_train(True) # Caller should handle this if needed
    
    if get_confusion_matrix:
        return correct / float(total), conf_matrix, avg_loss
    return correct / float(total), avg_loss

def get_dataloader_split_test(args, dataidxs, test_bs=32):
    dataset, datadir, download = args.dataset, args.datadir, args.download_data
    if dataset == 'cifar10':
        dl_obj = CIFAR10_truncated
        transform_test=transforms.Compose(
            [vision.ToTensor(),
            vision.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), is_hwc=False)])
        test_ds = dl_obj(datadir, dataidxs=dataidxs, train=False, transform=transform_test, download=download)
        
        # Create MindSpore GeneratorDataset
        # We need a generator function or class
        # test_ds implements __getitem__ and __len__, so it can be used directly if we wrap it
        
        def generator():
            for i in range(len(test_ds)):
                yield test_ds[i]
                
        # Or simpler: ds.GeneratorDataset(test_ds, column_names=["data", "label"])
        # GeneratorDataset accepts an iterable object that implements __getitem__ and __len__
        
        test_dl = ds.GeneratorDataset(test_ds, column_names=["data", "label"], shuffle=False)
        test_dl = test_dl.batch(test_bs)
        
    return test_dl

def get_dataloader(args, dataidxs=None, test_bs=32):
    dataset, datadir, train_bs, download = args.dataset, args.datadir, args.batch_size, args.download_data
    
    # Define transforms
    # Note: ToTensor produces (C, H, W). Normalize expects (C, H, W) if is_hwc=False (default is True?)
    # vision.Normalize default is_hwc=True. But ToTensor output is CHW.
    # So we MUST set is_hwc=False for Normalize after ToTensor.
    
    if dataset in ('cifar10', 'cifar100'):
        if dataset == 'cifar10':
            dl_obj = CIFAR10_truncated
            transform_train=transforms.Compose([
                vision.ToPIL(), # Convert back to PIL for RandomCrop/Flip if input is numpy
                vision.RandomCrop(32, padding=4),
                vision.RandomHorizontalFlip(),
                vision.ToTensor(),
                vision.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), is_hwc=False)])

            transform_test=transforms.Compose(
            [vision.ToTensor(),
             vision.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), is_hwc=False)])

        elif dataset == 'cifar100':
            dl_obj = CIFAR100_truncated
            
            # Mean/Std for CIFAR100
            mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
            std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

            transform_train = transforms.Compose([
                vision.ToPIL(),
                vision.RandomCrop(32, padding=4),
                vision.RandomHorizontalFlip(),
                vision.RandomRotation(15),
                vision.ToTensor(),
                vision.Normalize(mean, std, is_hwc=False)
            ])
            transform_test = transforms.Compose([
                vision.ToTensor(),
                vision.Normalize(mean, std, is_hwc=False)])
                
        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=download)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=download)
        
        train_dl = ds.GeneratorDataset(train_ds, column_names=["data", "label"], shuffle=True)
        train_dl = train_dl.batch(train_bs, drop_remainder=True)
        
        test_dl = ds.GeneratorDataset(test_ds, column_names=["data", "label"], shuffle=False)
        test_dl = test_dl.batch(test_bs)

    elif dataset == 'cinic10':
        dl_obj = ImageFolder_CINIC10
        transform_train=transforms.Compose([
            vision.RandomCrop(32, padding=4),
            vision.RandomHorizontalFlip(),
            vision.ToTensor(),
            vision.Normalize((0.47889522, 0.47227842, 0.43047404), (0.24205776, 0.23828046, 0.25874835), is_hwc=False)])

        transform_test=transforms.Compose(
        [vision.ToTensor(),
            vision.Normalize((0.47889522, 0.47227842, 0.43047404), (0.24205776, 0.23828046, 0.25874835), is_hwc=False)])
        
        train_ds = dl_obj(datadir+'/train/', dataidxs=dataidxs, transform=transform_train)
        test_ds = dl_obj(datadir+'/sampled test/', transform=transform_test)
        
        train_dl = ds.GeneratorDataset(train_ds, column_names=["data", "label"], shuffle=True)
        train_dl = train_dl.batch(train_bs, drop_remainder=True)
        
        test_dl = ds.GeneratorDataset(test_ds, column_names=["data", "label"], shuffle=False)
        test_dl = test_dl.batch(test_bs)

    elif dataset == 'ham10000':
        dl_obj = ImageFolder_HAM10000
        transform_train=transforms.Compose([
            vision.Resize((56, 56)),
            vision.RandomHorizontalFlip(),
            vision.RandomVerticalFlip(),
            vision.RandomRotation(20),
            vision.RandomColorAdjust(brightness=0.1, contrast=0.1, hue=0.1),
            vision.ToTensor(),
            vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], is_hwc=False)])
        transform_test=transforms.Compose([
            vision.Resize((56, 56)),
            vision.ToTensor(),
            vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], is_hwc=False)])

        train_ds = dl_obj(datadir+'/train_new/', dataidxs=dataidxs, transform=transform_train)
        test_ds = dl_obj(datadir+'/val_new/', transform=transform_test)
        
        train_dl = ds.GeneratorDataset(train_ds, column_names=["data", "label"], shuffle=True)
        train_dl = train_dl.batch(train_bs, drop_remainder=True)
        
        test_dl = ds.GeneratorDataset(test_ds, column_names=["data", "label"], shuffle=False)
        test_dl = test_dl.batch(test_bs)

    elif dataset == 'fmnist':
        transform_train = transforms.Compose([
            vision.ToTensor(),
            vision.Normalize((0.5,), (0.5,), is_hwc=False),
        ])
        transform_test = transforms.Compose([
            vision.ToTensor(),
            vision.Normalize((0.5,), (0.5,), is_hwc=False),
        ])
        train_ds = FashionMNIST_truncated(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=download)
        test_ds = FashionMNIST_truncated(datadir, train=False, transform=transform_test, download=download)
        
        train_dl = ds.GeneratorDataset(train_ds, column_names=["data", "label"], shuffle=True)
        train_dl = train_dl.batch(train_bs, drop_remainder=True)
        
        test_dl = ds.GeneratorDataset(test_ds, column_names=["data", "label"], shuffle=False)
        test_dl = test_dl.batch(test_bs)

    return train_dl, test_dl