import os
import logging
import copy
import tensorflow as tf
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
        # TensorFlow models automatically use GPU if available
        # net.build(input_shape=(None, 32, 32, 3))
        dummy_input = tf.zeros([1, 32, 32, 3])
        _ = net(dummy_input, training=True)
        nets[net_i] = net
        weights = net.get_weights()
        print("Number of weight tensors:", len(weights))

        print("weights:", weights[1][:10])
        # exit()
    return nets

def load_cifar10_data(datadir, download):
    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=download, transform=None)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=download, transform=None)
    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target
    return (X_train, y_train, X_test, y_test)

def load_cinic10_data(datadir):
    xray_train_ds = ImageFolder_CINIC10(datadir+'/train/', transform=None)
    xray_test_ds = ImageFolder_CINIC10(datadir+'/sampled test/', transform=None)

    X_train, y_train = np.array([s[0] for s in xray_train_ds.samples]), np.array([int(s[1]) for s in xray_train_ds.samples])
    X_test, y_test = np.array([s[0] for s in xray_test_ds.samples]), np.array([int(s[1]) for s in xray_test_ds.samples])
    return (X_train, y_train, X_test, y_test)

def load_HAM10000_data(datadir):
    xray_train_ds = ImageFolder_HAM10000(datadir+'/train_new/', transform=None)
    xray_test_ds = ImageFolder_HAM10000(datadir+'/val_new/', transform=None)

    X_train, y_train = np.array([s[0] for s in xray_train_ds.samples]), np.array([int(s[1]) for s in xray_train_ds.samples])
    X_test, y_test = np.array([s[0] for s in xray_test_ds.samples]), np.array([int(s[1]) for s in xray_test_ds.samples])
    return (X_train, y_train, X_test, y_test)

def load_cifar100_data(datadir, download):
    cifar100_train_ds = CIFAR100_truncated(datadir, train=True, download=download, transform=None)
    cifar100_test_ds = CIFAR100_truncated(datadir, train=False, download=download, transform=None)

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.target
    return (X_train, y_train, X_test, y_test)

def load_fmnist_data(datadir, download):
    fmnist_train_ds = FashionMNIST_truncated(datadir, train=True, download=download, transform=None)
    fmnist_test_ds = FashionMNIST_truncated(datadir, train=False, download=download, transform=None)

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
        _, _, _, y_test = load_cinic10_data(datadir, download)
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
    dataset, datadir, partition, n_parties, beta, n_niid_parties, global_imbalanced, download = args.dataset, args.datadir, \
        args.partition, args.n_parties, args.beta, args.n_niid_parties, args.train_global_imb, args.download_data
    if dataset == 'cifar10':
        _, y_train, _, _ = load_cifar10_data(datadir, download)
    elif dataset == 'cifar100':
        _, y_train, _, _ = load_cifar100_data(datadir, download)
    elif dataset == 'fmnist':
        _, y_train, _, _ = load_fmnist_data(datadir, download)
    elif dataset == 'cinic10':
        _, y_train, _, _ = load_cinic10_data(datadir, download)
    elif dataset == 'ham10000':
        _, y_train, _, _ = load_HAM10000_data(datadir, download)
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
                print(((20*i+j)%100)*num_sample_per_class+num_per_shard*(i//5)*5,((20*i+j)%100)*num_sample_per_class+num_per_shard*(i//5+1)*5)
            np.random.shuffle(net_dataidx_map[i])
            net_dataidx_map[i] = list(net_dataidx_map[i])

        # Partition of iid clients
        for i in range(num_non_iid_client,n_parties):
            for j in range(num_classes):
                net_dataidx_map[i] = np.concatenate(
                        (net_dataidx_map[i], idxs[j*num_sample_per_class+num_per_shard*5*(num_non_iid_client//5)+num_per_shard*(i-num_non_iid_client): \
                                                    j*num_sample_per_class+num_per_shard*5*(num_non_iid_client//5)+num_per_shard*(i-num_non_iid_client+1)]), axis=0)
                print(j*num_sample_per_class+num_per_shard*5*(num_non_iid_client//5)+num_per_shard*(i-num_non_iid_client),j*num_sample_per_class+num_per_shard*5*(num_non_iid_client//5)+num_per_shard*(i-num_non_iid_client+1))
            np.random.shuffle(net_dataidx_map[i])
            net_dataidx_map[i] = list(net_dataidx_map[i])

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    return (net_dataidx_map, traindata_cls_counts)

def set_client_from_params(mdl, params):
    weights = mdl.get_weights()
    idx = 0
    new_weights = []
    for w in weights:
        length = len(w.reshape(-1))
        new_w = tf.constant(params[idx:idx+length].reshape(w.shape), dtype=tf.float32)
        new_weights.append(new_w)
        idx += length
    mdl.set_weights(new_weights)
    return mdl

def get_mdl_params(model_list, n_par=None):
    if n_par==None:
        exp_mdl = model_list[0]
        n_par = 0
        for w in exp_mdl.get_weights():
            n_par += len(w.reshape(-1))

    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, mdl in enumerate(model_list):
        idx = 0
        for w in mdl.get_weights():
            temp = w.numpy().reshape(-1)
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)

def compute_accuracy(model, dataloader, get_confusion_matrix=False, device="cpu", multiloader=False):
    was_training = model.trainable
    if was_training:
        model.trainable = False

    correct, total = 0, 0
    true_labels_list, pred_labels_list = np.array([]), np.array([])
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss_collector = []
    if multiloader:
        for loader in dataloader:
            for batch_idx, (x, target) in enumerate(loader):
                if isinstance(x, np.ndarray):
                    x = tf.constant(x, dtype=tf.float32)
                if isinstance(target, np.ndarray):
                    target = tf.constant(target, dtype=tf.int64)
                _, _, out = model(x, training=False)
                if len(target.shape) == 0 or target.shape[0] == 1:
                    out = tf.expand_dims(out, 0)
                    if len(target.shape) == 0:
                        target = tf.expand_dims(target, 0)
                    loss = criterion(target, out)
                else:
                    loss = criterion(target, out)

                pred_label = tf.argmax(out, axis=1)
                loss_collector.append(float(loss.numpy()))
                total += x.shape[0]
                correct += int(tf.reduce_sum(tf.cast(pred_label == target, tf.int32)).numpy())

                pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                true_labels_list = np.append(true_labels_list, target.numpy())
        avg_loss = sum(loss_collector) / len(loss_collector) if loss_collector else 0.0
    else:
        for batch_idx, (x, target) in enumerate(dataloader):
            # if isinstance(x, np.ndarray):
            #     x = tf.constant(x, dtype=tf.float32)
            if isinstance(target, np.ndarray):
                target = tf.constant(target, dtype=tf.int64)
            _, _, out = model(x, training=False)

            # print(out.shape, target.shape)
            # print(out[2])
            # exit()
            loss = criterion(target, out)
            pred_label = tf.argmax(out, axis=1)
            # print(pred_label.numpy())
            # print(target.numpy())
            loss_collector.append(float(loss.numpy()))
            total += x.shape[0]
            # correct += int(tf.reduce_sum(tf.cast(pred_label == target, tf.int32)).numpy())
            # 统一类型为 int64 再比较
            correct += int(tf.reduce_sum(
                tf.cast(
                tf.equal(tf.cast(pred_label, tf.int64), tf.cast(target, tf.int64)), tf.int32)
            ).numpy())
            # print(correct)
            # exit()
            pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
            true_labels_list = np.append(true_labels_list, target.numpy())
        avg_loss = sum(loss_collector) / len(loss_collector) if loss_collector else 0.0

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)
    if was_training:
        model.trainable = True
    if get_confusion_matrix:
        return correct / float(total), conf_matrix, avg_loss
    return correct / float(total), avg_loss

def _preprocess_cifar10_train(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    x = tf.image.random_crop(x, [32, 32, 3])
    x = tf.image.random_flip_left_right(x)
    x = tf.image.per_image_standardization(x)
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    x = (x - mean) / std
    return x, y

def _preprocess_cifar10_test(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    x = (x - mean) / std
    return x, y

def _preprocess_cifar100_train(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    x = tf.image.random_crop(x, [32, 32, 3])
    x = tf.image.random_flip_left_right(x)
    mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
    std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
    x = (x - mean) / std
    return x, y

def _preprocess_cifar100_test(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
    std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
    x = (x - mean) / std
    return x, y

def _preprocess_fmnist_train(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    x = (x - 0.5) / 0.5
    if len(x.shape) == 2:
        x = tf.expand_dims(x, -1)
    return x, y

def _preprocess_fmnist_test(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    x = (x - 0.5) / 0.5
    if len(x.shape) == 2:
        x = tf.expand_dims(x, -1)
    return x, y

def get_dataloader_split_test(args, dataidxs, test_bs=32):
    dataset, datadir, download = args.dataset, args.datadir, args.download_data
    if dataset == 'cifar10':
        dl_obj = CIFAR10_truncated
        test_ds = dl_obj(datadir, dataidxs=dataidxs, train=False, transform=None, download=download)
        test_data = tf.data.Dataset.from_tensor_slices((test_ds.data, test_ds.target))
        test_data = test_data.map(_preprocess_cifar10_test)
        test_dl = test_data.batch(test_bs).prefetch(tf.data.AUTOTUNE)
    return test_dl

def get_dataloader(args, dataidxs=None, test_bs=32):
    dataset, datadir, train_bs, download = args.dataset, args.datadir, args.batch_size, args.download_data
    if dataset in ('cifar10', 'cifar100'):
        if dataset == 'cifar10':
            dl_obj = CIFAR10_truncated
            train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=None, download=download)
            test_ds = dl_obj(datadir, train=False, transform=None, download=download)
            train_data = tf.data.Dataset.from_tensor_slices((train_ds.data, train_ds.target))
            test_data = tf.data.Dataset.from_tensor_slices((test_ds.data, test_ds.target))
            train_data = train_data.map(_preprocess_cifar10_train)
            test_data = test_data.map(_preprocess_cifar10_test)
            train_dl = train_data.shuffle(10000).batch(train_bs, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
            test_dl = test_data.batch(test_bs).prefetch(tf.data.AUTOTUNE)

        elif dataset == 'cifar100':
            dl_obj = CIFAR100_truncated
            train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=None, download=download)
            test_ds = dl_obj(datadir, train=False, transform=None, download=download)
            train_data = tf.data.Dataset.from_tensor_slices((train_ds.data, train_ds.target))
            test_data = tf.data.Dataset.from_tensor_slices((test_ds.data, test_ds.target))
            train_data = train_data.map(_preprocess_cifar100_train)
            test_data = test_data.map(_preprocess_cifar100_test)
            train_dl = train_data.shuffle(10000).batch(train_bs, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
            test_dl = test_data.batch(test_bs).prefetch(tf.data.AUTOTUNE)

    elif dataset == 'cinic10':
        dl_obj = ImageFolder_CINIC10
        train_ds = dl_obj(datadir+'/train/', dataidxs=dataidxs, transform=None)
        test_ds = dl_obj(datadir+'/sampled test/', transform=None)
        # For ImageFolder datasets, need to load images
        def load_and_preprocess_train(path, label):
            img = tf.io.read_file(path)
            img = tf.image.decode_image(img, channels=3)
            img = tf.cast(img, tf.float32) / 255.0
            img = tf.image.resize_with_pad(img, 36, 36)
            img = tf.image.random_crop(img, [32, 32, 3])
            img = tf.image.random_flip_left_right(img)
            mean = [0.47889522, 0.47227842, 0.43047404]
            std = [0.24205776, 0.23828046, 0.25874835]
            img = (img - mean) / std
            return img, label
        def load_and_preprocess_test(path, label):
            img = tf.io.read_file(path)
            img = tf.image.decode_image(img, channels=3)
            img = tf.cast(img, tf.float32) / 255.0
            img = tf.image.resize(img, [32, 32])
            mean = [0.47889522, 0.47227842, 0.43047404]
            std = [0.24205776, 0.23828046, 0.25874835]
            img = (img - mean) / std
            return img, label
        train_data = tf.data.Dataset.from_tensor_slices((train_ds.samples[:, 0], train_ds.samples[:, 1]))
        test_data = tf.data.Dataset.from_tensor_slices((test_ds.samples[:, 0], test_ds.samples[:, 1]))
        train_data = train_data.map(load_and_preprocess_train)
        test_data = test_data.map(load_and_preprocess_test)
        train_dl = train_data.shuffle(10000).batch(train_bs, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        test_dl = test_data.batch(test_bs).prefetch(tf.data.AUTOTUNE)

    elif dataset == 'ham10000':
        dl_obj = ImageFolder_HAM10000
        train_ds = dl_obj(datadir+'/train_new/', dataidxs=dataidxs, transform=None)
        test_ds = dl_obj(datadir+'/val_new/', transform=None)
        def load_and_preprocess_train(path, label):
            img = tf.io.read_file(path)
            img = tf.image.decode_image(img, channels=3)
            img = tf.image.resize(img, [56, 56])
            img = tf.cast(img, tf.float32) / 255.0
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            img = (img - mean) / std
            return img, label
        def load_and_preprocess_test(path, label):
            img = tf.io.read_file(path)
            img = tf.image.decode_image(img, channels=3)
            img = tf.image.resize(img, [56, 56])
            img = tf.cast(img, tf.float32) / 255.0
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            img = (img - mean) / std
            return img, label
        train_data = tf.data.Dataset.from_tensor_slices((train_ds.samples[:, 0], train_ds.samples[:, 1]))
        test_data = tf.data.Dataset.from_tensor_slices((test_ds.samples[:, 0], test_ds.samples[:, 1]))
        train_data = train_data.map(load_and_preprocess_train)
        test_data = test_data.map(load_and_preprocess_test)
        train_dl = train_data.shuffle(10000).batch(train_bs, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        test_dl = test_data.batch(test_bs).prefetch(tf.data.AUTOTUNE)

    elif dataset == 'fmnist':
        train_ds = FashionMNIST_truncated(datadir, dataidxs=dataidxs, train=True, transform=None, download=download)
        test_ds = FashionMNIST_truncated(datadir, train=False, transform=None, download=download)
        train_data = tf.data.Dataset.from_tensor_slices((train_ds.data, train_ds.target))
        test_data = tf.data.Dataset.from_tensor_slices((test_ds.data, test_ds.target))
        train_data = train_data.map(_preprocess_fmnist_train)
        test_data = test_data.map(_preprocess_fmnist_test)
        train_dl = train_data.shuffle(10000).batch(train_bs, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        test_dl = test_data.batch(test_bs).prefetch(tf.data.AUTOTUNE)

    return train_dl, test_dl