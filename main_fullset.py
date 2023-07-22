"""Our codes are adapted from codes in Model-Contrastive Federated Learning"""

from importlib_metadata import distribution
import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import logging
import os
import copy
import datetime
import random
import sys
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from model import *
from utils import *

from distribution_aware.utils import get_distribution_difference


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='simple-cnn', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='noniid', help='the data partitioning strategy')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=10, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=10, help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg',
                        help='communication strategy: fedavg/fedprox')
    parser.add_argument('--comm_round', type=int, default=100, help='number of maximum communication roun')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox or moon')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--local_max_epoch', type=int, default=100, help='the number of epoch for local optimal training')
    parser.add_argument('--model_buffer_size', type=int, default=1, help='store how many previous models for contrastive loss')
    parser.add_argument('--pool_option', type=str, default='FIFO', help='FIFO or BOX')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='how many clients are sampled in each round')
    parser.add_argument('--load_model_file', type=str, default=None, help='the model to load as global model')
    parser.add_argument('--load_pool_file', type=str, default=None, help='the old model pool path to load')
    parser.add_argument('--load_model_round', type=int, default=None, help='how many rounds have executed for the loaded model')
    parser.add_argument('--load_first_net', type=int, default=1, help='whether load the first net as old net or not')
    parser.add_argument('--normal_model', type=int, default=0, help='use normal model or aggregate model')
    parser.add_argument('--loss', type=str, default='contrastive')
    parser.add_argument('--save_model',type=int,default=0)
    parser.add_argument('--use_project_head', type=int, default=0)
    parser.add_argument('--server_momentum', type=float, default=0, help='the server momentum (FedAvgM)')

    # Experiments on FedDisco
    parser.add_argument('--n_niid_parties', type=int, default=5, help='number of niid workers')         # for non-iid-4
    parser.add_argument('--distribution_aware', type=str, default='not', help='Types of distribution aware e.g. division')
    parser.add_argument('--measure_difference', type=str, default='l2', help='How to measure difference. e.g. only_iid, cosine')
    parser.add_argument('--difference_operation', type=str, default='square', help='Conduct operation on difference. e.g. square or cube')
    parser.add_argument('--disco_a', type=float, default=0.1, help='Under sub mode, n_k-disco_a*d_k+disco_b')
    parser.add_argument('--disco_b', type=float, default=0.0)
    parser.add_argument('--disco_partition', type=int, default=0, help='Whether to apply model partition, 0 means not. 1 means apply')
    parser.add_argument('--global_imbalanced', action='store_true') # imbalanced global dataset

    args = parser.parse_args()
    return args


def init_nets(net_configs, n_parties, args, device='cpu'):
    nets = {net_i: None for net_i in range(n_parties)}
    n_classes = 10
    
    for net_i in range(n_parties):
        net = ModelFedCon_noheader(args.model, args.out_dim, n_classes, net_configs, args.dataset)
        if device == 'cpu':
            net.to(device)
        else:
            net = net.cuda()
        nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type


def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device="cpu"):
    net.cuda()
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().cuda()

    cnt = 0
    
    for epoch in range(epochs):     
        epoch_loss_collector = []    
        for batch_idx, (x, target) in enumerate(train_dataloader):  
            x, target = x.cuda(), target.cuda()
            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            _,_,out = net(x)
            loss = criterion(out, target)
            
            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
    logger.info('>> Test accuracy: %f' % test_acc)
    net.to('cpu')

    logger.info(' ** Training complete **')
    return test_acc


def local_train_net(nets, args, net_dataidx_map, train_dl=None, test_dl=None, global_model = None, prev_model_pool = None, server_c = None, clients_c = None, round=None, device="cpu", current_round=0):
    avg_acc = 0.0
    acc_list = []
    if global_model:
        global_model.cuda()
    if server_c:
        server_c.cuda()
        server_c_collector = list(server_c.cuda().parameters())
        new_server_c_collector = copy.deepcopy(server_c_collector)
    for net_id, net in nets.items():   
        dataidxs = net_dataidx_map[net_id]
        train_dl_local=train_dl[net_id]   
        n_epoch = args.epochs

        if args.alg == 'fedavg':
            testacc = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args,
                                        device=device)

        print("Training network %s, n_training: %d, final test acc %f." % (str(net_id), len(dataidxs), testacc))
        acc_list.append(testacc)

    if global_model:
        global_model.to('cpu')
    if server_c:
        for param_index, param in enumerate(server_c.parameters()):
            server_c_collector[param_index] = new_server_c_collector[param_index]
        server_c.to('cpu')
    return acc_list


if __name__ == '__main__':
    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    now_time = datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    print(now_time)
    vis_dir = os.path.join('visualize/exp_vis', now_time)
    os.mkdir(vis_dir)

    dataset_logdir = os.path.join(args.logdir, args.dataset)
    mkdirs(dataset_logdir)

    writer = SummaryWriter(os.path.join(args.logdir, 'board'))
    
    if args.log_file_name is None:
        argument_path = 'experiment_arguments-%s.json' % (now_time)
    else:
        argument_path = args.log_file_name + '.json'
    with open(os.path.join(dataset_logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    device = torch.device(args.device)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (now_time)
    log_path = args.log_file_name + '.log'

    logging.basicConfig(
        filename=os.path.join(dataset_logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.INFO, filemode='w')

    logger = logging.getLogger()
    logger.info(device)

    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)

    # data partition
    logger.info("Partitioning data")
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, dataset_logdir, args.partition, args.n_parties, beta=args.beta, n_niid_parties=args.n_niid_parties, global_imbalanced=args.global_imbalanced)

    n_party_per_round = int(args.n_parties * args.sample_fraction)
    party_list = [i for i in range(args.n_parties)]
    party_list_rounds = []
    if n_party_per_round != args.n_parties:
        for i in range(args.comm_round):
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:
        for i in range(args.comm_round):
            party_list_rounds.append(party_list)

    n_classes = len(np.unique(y_train))

    # get testing dataloader
    train_dl_global, test_dl, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                               args.datadir,
                                                                               args.batch_size,
                                                                               32)

    print("len train_dl_global:", len(train_ds_global))
    train_dl=None
    data_size = len(test_ds_global)
    # net initialization
    logger.info("Initializing nets")
    nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.n_parties, args, device='cpu')
    global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 1, args, device='cpu')
    global_model = global_models[0]

    n_comm_rounds = args.comm_round
    if args.load_model_file and args.alg != 'plot_visual':
        global_model.load_state_dict(torch.load(args.load_model_file))
        n_comm_rounds -= args.load_model_round

    if args.server_momentum:
        moment_v = copy.deepcopy(global_model.state_dict())
        for key in moment_v:
            moment_v[key] = 0
    print(device)
    # get training dataloader for each client
    train_local_dls=[]  
    for net_id, net in nets.items():
        dataidxs = net_dataidx_map[net_id]
        train_dl_local, _, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs)
        train_local_dls.append(train_dl_local)

    best_test_acc=0
    record_test_acc_list = []
    acc_dir = os.path.join(dataset_logdir, 'acc_list')
    if not os.path.exists(acc_dir):
        os.mkdir(acc_dir)

    acc_path = os.path.join(dataset_logdir, f'acc_list/{now_time}.npy')


    if args.alg == 'fedavg':
        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            print("In communication round:" + str(round))
            party_list_this_round = party_list_rounds[round]
            if args.sample_fraction<1.0:
                print(f'Clients this round : {party_list_this_round}')
            # Model Initialization
            global_w = global_model.state_dict()
            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)
            
            # Local Update
            test_acc_list = local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_local_dls, test_dl=test_dl, device=device)
            
            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]
            if round==0 or args.sample_fraction<1.0:
                print(f'Dataset size weight : {fed_avg_freqs}')

            # calculate the discrepancy 
            distribution_difference = get_distribution_difference(traindata_cls_counts, participation_clients=party_list_this_round, metric=args.measure_difference)

            if np.sum(distribution_difference) == 0:
                distribution_difference = np.array([0 for _ in range(len(distribution_difference))])
            else:
                distribution_difference = distribution_difference / np.sum(distribution_difference)     # normalize. (some metrics make the difference value larger than 1.0)
            if args.difference_operation=='linear':
                pass
            elif args.difference_operation=='square':
                distribution_difference = np.power(distribution_difference,2)
            elif args.difference_operation=='cube':
                distribution_difference = np.power(distribution_difference,3)
            else:
                raise NotImplementedError
            if round==0 or args.sample_fraction<1.0:
                print(f'Distribution_difference : {distribution_difference}')

            # adjusting the aggregation weight
            if args.distribution_aware != 'not':
                a=args.disco_a
                b=args.disco_b
                tmp = fed_avg_freqs - a*distribution_difference + b
                if np.sum(tmp>0)>0:         # ensure not all elements are smaller than 0
                    fed_avg_freqs = np.copy(tmp)
                    fed_avg_freqs[fed_avg_freqs<0.0]=0.0
                total_normalizer = sum([fed_avg_freqs[r] for r in range(len(party_list_this_round))])
                fed_avg_freqs = [fed_avg_freqs[r] / total_normalizer for r in range(len(party_list_this_round))]
                if round==0 or args.sample_fraction<1.0:
                    print(f'Disco Aggregation Weights : {fed_avg_freqs}')

            # Model Aggregation
            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()  
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]

            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1-args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]

            global_model.load_state_dict(global_w)

            logger.info('global n_test: %d' % len(test_dl))
            global_model.cuda()
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device, dataset = args.dataset)
            record_test_acc_list.append(test_acc)
            writer.add_scalar('testing accuracy',
                            test_acc,
                            round)
            if(best_test_acc<test_acc):
                best_test_acc=test_acc
                logger.info('New Best best test acc:%f'% test_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            logger.info('>> Global Model Best accuracy: %f' % best_test_acc)
            print('>> Global Model Test accuracy: %f, Best: %f' % (test_acc, best_test_acc))
           
            mkdirs(args.modeldir+'fedavg/')
            global_model.to('cpu')

            torch.save(global_model.state_dict(), args.modeldir+'fedavg/'+'globalmodel'+args.log_file_name+'.pth')
            torch.save(nets[0].state_dict(), args.modeldir+'fedavg/'+'localmodel0'+args.log_file_name+'.pth')
        np.save(acc_path, np.array(record_test_acc_list))
    
    print('>> Global Model Best accuracy: %f' % best_test_acc)
    print(args)
    print(now_time)