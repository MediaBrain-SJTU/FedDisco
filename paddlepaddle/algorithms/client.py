import copy
import paddle
import paddle.nn as nn
import paddle.optimizer as optim
from utils.utils import compute_accuracy


def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device="cpu",
              logger=None):
    # PaddlePaddle设备管理
    if device == "cuda" or device.startswith("gpu"):
        paddle.set_device('gpu:0')
    else:
        paddle.set_device('cpu')

    logger.info('Training network %s' % str(net_id))

    # 在PaddlePaddle中，不需要filter，直接使用parameters()
    if args_optimizer == 'adam':
        optimizer = optim.Adam(parameters=net.parameters(), learning_rate=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(parameters=net.parameters(), learning_rate=lr, weight_decay=args.reg,
                               beta1=0.9, beta2=0.999, epsilon=1e-8)
    elif args_optimizer == 'sgd':
        optimizer = optim.Momentum(parameters=net.parameters(), learning_rate=lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            # 数据转换：如果数据不是paddle.Tensor，则转换为tensor
            if not isinstance(x, paddle.Tensor):
                x = paddle.to_tensor(x)
            if not isinstance(target, paddle.Tensor):
                target = paddle.to_tensor(target)

            # 确保target是int64类型
            target = target.astype('int64')

            _, _, out = net(x)
            loss = criterion(out, target)

            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            epoch_loss_collector.append(float(loss.item()))

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
    logger.info('>> Test accuracy: %f' % test_acc)
    return test_acc


def train_net_scaffold(net_id, net, global_model, c_local, c_global, train_dataloader, test_dataloader, epochs, lr,
                       args_optimizer, args, device="cpu", logger=None):
    logger.info('Training network %s' % str(net_id))

    # 设置设备
    if device == "cuda" or device.startswith("gpu"):
        paddle.set_device('gpu:0')
    else:
        paddle.set_device('cpu')

    if args_optimizer == 'adam':
        optimizer = optim.Adam(parameters=net.parameters(), learning_rate=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(parameters=net.parameters(), learning_rate=lr, weight_decay=args.reg,
                               beta1=0.9, beta2=0.999, epsilon=1e-8)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(parameters=net.parameters(), learning_rate=lr, momentum=0.9, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss()

    cnt = 0
    c_global_para = c_global.state_dict()
    c_local_para = c_local.state_dict()

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            # 数据转换
            if not isinstance(x, paddle.Tensor):
                x = paddle.to_tensor(x)
            if not isinstance(target, paddle.Tensor):
                target = paddle.to_tensor(target)

            optimizer.clear_grad()
            target = target.astype('int64')

            _, _, out = net(x)
            loss = criterion(out, target)

            loss.backward()
            optimizer.step()

            net_para = net.state_dict()
            for key in net_para:
                net_para[key] = net_para[key] - args.lr * (c_global_para[key] - c_local_para[key])
            net.set_state_dict(net_para)
            cnt += 1
            epoch_loss_collector.append(float(loss.item()))

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    c_new_para = c_local.state_dict()
    c_delta_para = copy.deepcopy(c_local.state_dict())
    global_model_para = global_model.state_dict()
    net_para = net.state_dict()
    for key in net_para:
        c_new_para[key] = c_new_para[key] - c_global_para[key] + (global_model_para[key] - net_para[key]) / (
                    cnt * args.lr)
        c_delta_para[key] = c_new_para[key] - c_local_para[key]
    c_local.set_state_dict(c_new_para)

    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
    logger.info('>> Test accuracy: %f' % test_acc)
    return test_acc, c_delta_para


def train_net_fedprox(net_id, net, global_net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, args,
                      device="cpu", logger=None):
    logger.info('Training network %s' % str(net_id))

    # 设置设备
    if device == "cuda" or device.startswith("gpu"):
        paddle.set_device('gpu:0')
    else:
        paddle.set_device('cpu')

    if args_optimizer == 'adam':
        optimizer = optim.Adam(parameters=net.parameters(), learning_rate=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(parameters=net.parameters(), learning_rate=lr, weight_decay=args.reg,
                               beta1=0.9, beta2=0.999, epsilon=1e-8)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(parameters=net.parameters(), learning_rate=lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss()

    global_weight_collector = list(global_net.parameters())

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            # 数据转换
            if not isinstance(x, paddle.Tensor):
                x = paddle.to_tensor(x)
            if not isinstance(target, paddle.Tensor):
                target = paddle.to_tensor(target)

            optimizer.clear_grad()
            target = target.astype('int64')

            _, _, out = net(x)
            loss = criterion(out, target)

            # For fedprox
            fed_prox_reg = 0.0
            for param_index, param in enumerate(net.parameters()):
                fed_prox_reg += ((mu / 2) * paddle.norm(param - global_weight_collector[param_index], p=2) ** 2)
            loss += fed_prox_reg

            loss.backward()
            optimizer.step()
            epoch_loss_collector.append(float(loss.item()))

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
    return test_acc


def train_net_fedcon(net_id, net, global_net, previous_nets, train_dataloader, test_dataloader, epochs, lr,
                     args_optimizer, mu, temperature, args,
                     round, device="cpu", logger=None):
    logger.info('Training network %s' % str(net_id))

    # 设置设备
    if device == "cuda" or device.startswith("gpu"):
        paddle.set_device('gpu:0')
    else:
        paddle.set_device('cpu')

    if args_optimizer == 'adam':
        optimizer = optim.Adam(parameters=net.parameters(), learning_rate=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(parameters=net.parameters(), learning_rate=lr, weight_decay=args.reg,
                               beta1=0.9, beta2=0.999, epsilon=1e-8)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(parameters=net.parameters(), learning_rate=lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss()

    cos = nn.CosineSimilarity(axis=-1)

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            # 数据转换
            if not isinstance(x, paddle.Tensor):
                x = paddle.to_tensor(x)
            if not isinstance(target, paddle.Tensor):
                target = paddle.to_tensor(target)

            optimizer.clear_grad()
            target = target.astype('int64')

            _, pro1, out = net(x)
            _, pro2, _ = global_net(x)

            posi = cos(pro1, pro2)
            logits = posi.reshape([-1, 1])

            for previous_net in previous_nets:
                _, pro3, _ = previous_net(x)
                nega = cos(pro1, pro3)
                logits = paddle.concat([logits, nega.reshape([-1, 1])], axis=1)

            logits /= temperature
            labels = paddle.zeros([x.shape[0]], dtype='int64')

            loss2 = mu * criterion(logits, labels)
            loss1 = criterion(out, target)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            epoch_loss_collector.append(float(loss.item()))

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
    return test_acc


def local_train_net(nets, args, net_dataidx_map, train_dl=None, test_dl=None, global_model=None, prev_model_pool=None,
                    round=None, device="cpu", logger=None):
    acc_list = []

    # 设置设备
    if device == "cuda" or device.startswith("gpu"):
        paddle.set_device('gpu:0')
    else:
        paddle.set_device('cpu')

    if global_model:
        # 在PaddlePaddle中，模型会自动在设置的设备上运行
        pass

    for net_id, net in nets.items():
        dataidxs = net_dataidx_map[net_id]
        train_dl_local = train_dl[net_id]
        n_epoch = args.epochs

        if args.alg == 'fedavg' or args.alg == 'fednova':
            testacc = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args,
                                device=device, logger=logger)
        elif args.alg == 'fedprox':
            testacc = train_net_fedprox(net_id, net, global_model, train_dl_local, test_dl, n_epoch, args.lr,
                                        args.optimizer, args.mu, args, device=device, logger=logger)
        elif args.alg == 'moon':
            prev_models = []
            for i in range(len(prev_model_pool)):
                prev_models.append(prev_model_pool[i][net_id])
            testacc = train_net_fedcon(net_id, net, global_model, prev_models, train_dl_local, test_dl, n_epoch,
                                       args.lr,
                                       args.optimizer, args.mu, args.temperature, args, round, device=device,
                                       logger=logger)
        print("Training network %s, n_training: %d, final test acc %f." % (str(net_id), len(dataidxs), testacc))
        acc_list.append(testacc)

    return acc_list


def local_train_net_scaffold(nets, global_model, c_nets, c_global, args, net_dataidx_map, train_dl=None, test_dl=None,
                             device="cpu", logger=None):
    global_model_para = global_model.state_dict()
    total_delta = {}
    for key in global_model_para:
        total_delta[key] = paddle.zeros_like(global_model_para[key])

    # 设置设备
    if device == "cuda" or device.startswith("gpu"):
        paddle.set_device('gpu:0')
    else:
        paddle.set_device('cpu')

    for idx, (net_id, net) in enumerate(nets.items()):
        dataidxs = net_dataidx_map[net_id]
        train_dl_local = train_dl[net_id]
        n_epoch = args.epochs

        testacc, c_delta_para = train_net_scaffold(net_id, net, global_model, c_nets[net_id], c_global, train_dl_local,
                                                   test_dl, n_epoch, args.lr, args.optimizer, args, device=device,
                                                   logger=logger)

        for key in total_delta:
            total_delta[key] += c_delta_para[key] / len(c_nets)
        print("Training network %s, n_training: %d, final test acc %f." % (str(net_id), len(dataidxs), testacc))

    c_global_para = c_global.state_dict()
    for key in c_global_para:
        # 确保类型匹配
        if c_global_para[key].dtype == 'int64' or str(c_global_para[key].dtype) == 'paddle.int64':
            total_delta[key] = total_delta[key].astype('int64')
        c_global_para[key] = c_global_para[key] + total_delta[key]
    c_global.set_state_dict(c_global_para)

    nets_list = list(nets.values())
    return nets_list

