import copy
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import value_and_grad, Tensor, ParameterTuple
from utils.utils import compute_accuracy

def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device="cpu", logger=None):
    # net.cuda() -> handled by context
    if logger:
        logger.info('Training network %s' % str(net_id))

    # Optimizer
    # MindSpore optimizers need a list of parameters
    params = net.trainable_params()
    if args_optimizer == 'adam':
        optimizer = nn.Adam(params, learning_rate=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = nn.Adam(params, learning_rate=lr, weight_decay=args.reg, use_amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = nn.SGD(params, learning_rate=lr, momentum=0.9, weight_decay=args.reg)
    
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # Define forward function
    def forward_fn(data, label):

        logits = net(data)
        loss = criterion(logits, label)
        return loss, logits

    # Get gradient function
    grad_fn = value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    net.set_train(True)
    
    for epoch in range(epochs): 
        epoch_loss_collector = []    
        # train_dataloader is a MindSpore dataset object
        # create_tuple_iterator(num_epochs=1) is safer
        step = 0
        for data, label in train_dataloader.create_tuple_iterator(num_epochs=1):
            # data, label are Tensors
            # print(data.shape)
            # print(label.shape)
            data = data.squeeze(1)
            label = label.astype(mindspore.int32)
            
            (loss, _), grads = grad_fn(data, label)
            optimizer(grads)
            
            epoch_loss_collector.append(loss.asnumpy())
            step += 1
            
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        if logger:
            logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
    if logger:
        logger.info('>> Test accuracy: %f' % test_acc)
    return test_acc

def train_net_scaffold(net_id, net, global_model, c_local, c_global, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device="cpu", logger=None):
    if logger:
        logger.info('Training network %s' % str(net_id))

    params = net.trainable_params()
    if args_optimizer == 'adam':
        optimizer = nn.Adam(params, learning_rate=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = nn.Adam(params, learning_rate=lr, weight_decay=args.reg, use_amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = nn.SGD(params, learning_rate=lr, momentum=0.9, weight_decay=args.reg)
        
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    cnt = 0
    # Pre-compute difference between c_global and c_local for each param
    # c_global and c_local are models (Cells).
    # We need to map them to net params.
    # Assuming they have same structure and order.
    
    c_global_params = {p.name: p for p in c_global.trainable_params()}
    c_local_params = {p.name: p for p in c_local.trainable_params()}
    
    # We need to access these inside the loop.
    # To make it efficient, we can compute the difference tensors once if they are constant during local training?
    # Yes, c_global and c_local are constant during local update.
    
    c_diff = []
    for p in params:
        # Find corresponding c params. 
        # Note: param names might differ if models are different instances, but usually they match if created same way.
        # But here net_id is specific.
        # Let's assume strict ordering match.
        pass

    # Actually, we can just iterate by index if we trust the order.
    c_global_list = list(c_global.trainable_params())
    c_local_list = list(c_local.trainable_params())
    
    def forward_fn(data, label):
        logits = net(data)
        loss = criterion(logits, label)
        return loss, logits

    grad_fn = value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    net.set_train(True)

    for epoch in range(epochs):
        epoch_loss_collector = []
        for data, label in train_dataloader.create_tuple_iterator(num_epochs=1):
            label = label.astype(mindspore.int32)
            
            (loss, _), grads = grad_fn(data, label)
            
            # Scaffold correction: grad = grad + c_global - c_local
            # We need to modify grads tuple. Tuples are immutable, so create new list.
            new_grads = []
            for i, g in enumerate(grads):
                c_g = c_global_list[i]
                c_l = c_local_list[i]
                # g + c_g - c_l
                new_g = g + c_g - c_l
                new_grads.append(new_g)
            
            optimizer(tuple(new_grads))
            
            cnt += 1
            epoch_loss_collector.append(loss.asnumpy())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        if logger:
            logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    # Update c_local
    # c_new = c_local - c_global + (global_model - net) / (cnt * lr)
    # c_delta = c_new - c_local
    
    # We need to do this update on parameters.
    # c_local is a Cell.
    
    global_params = list(global_model.trainable_params())
    net_params = list(net.trainable_params())
    
    c_delta_para = {} # Store as dict to return? Or just list? Original returns dict.
    
    for i, c_l in enumerate(c_local_list):
        c_g = c_global_list[i]
        g_p = global_params[i]
        n_p = net_params[i]
        
        # Calculation
        # c_new = c_l - c_g + (g_p - n_p) / (cnt * lr)
        # But wait, original code:
        # c_new_para[key] = c_new_para[key] - c_global_para[key] + (global_model_para[key] - net_para[key]) / (cnt * args.lr)
        # c_delta_para[key] = c_new_para[key] - c_local_para[key]
        
        # So c_new = c_local - c_global + ...
        
        term = (g_p - n_p) / (cnt * lr)
        c_new_val = c_l - c_g + term
        c_delta_val = c_new_val - c_l
        
        # Update c_local
        c_l.set_data(c_new_val)
        
        # Store delta. Using param name as key to match original dict structure if possible.
        c_delta_para[c_l.name] = c_delta_val

    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
    if logger:
        logger.info('>> Test accuracy: %f' % test_acc)
    return test_acc, c_delta_para

def train_net_fedprox(net_id, net, global_net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, args,
                      device="cpu", logger=None):
    if logger:
        logger.info('Training network %s' % str(net_id))
    
    params = net.trainable_params()
    if args_optimizer == 'adam':
        optimizer = nn.Adam(params, learning_rate=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = nn.Adam(params, learning_rate=lr, weight_decay=args.reg, use_amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = nn.SGD(params, learning_rate=lr, momentum=0.9, weight_decay=args.reg)

    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # Capture global weights
    global_weight_collector = list(global_net.trainable_params())

    def forward_fn(data, label):
        logits = net(data)
        loss = criterion(logits, label)
        
        # FedProx regularization
        # (mu / 2) * ||w - w_t||^2
        fed_prox_reg = 0.0
        # We need to iterate current params.
        # Note: 'net' is captured.
        for i, param in enumerate(net.trainable_params()):
            diff = param - global_weight_collector[i]
            fed_prox_reg += (mu / 2) * ops.norm(diff) ** 2
            
        loss += fed_prox_reg
        return loss, logits

    grad_fn = value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
    
    net.set_train(True)

    for epoch in range(epochs):
        epoch_loss_collector = []
        for data, label in train_dataloader.create_tuple_iterator(num_epochs=1):
            label = label.astype(mindspore.int32)
            (loss, _), grads = grad_fn(data, label)
            optimizer(grads)
            epoch_loss_collector.append(loss.asnumpy())
            
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        if logger:
            logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
    return test_acc

def train_net_fedcon(net_id, net, global_net, previous_nets, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, temperature, args,
                      round, device="cpu", logger=None):
    if logger:
        logger.info('Training network %s' % str(net_id))

    params = net.trainable_params()
    if args_optimizer == 'adam':
        optimizer = nn.Adam(params, learning_rate=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = nn.Adam(params, learning_rate=lr, weight_decay=args.reg, use_amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = nn.SGD(params, learning_rate=lr, momentum=0.9, weight_decay=args.reg)

    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    cos = ops.CosineSimilarity(axis=-1)

    def forward_fn(data, label):
        # net returns h, h, y (features, features, output)
        # See model.py: return h, h, y
        _, pro1, out = net(data)
        _, pro2, _ = global_net(data)
        
        # Contrastive loss
        posi = cos(pro1, pro2)
        logits = posi.reshape(-1, 1)
        
        for previous_net in previous_nets:
            _, pro3, _ = previous_net(data)
            nega = cos(pro1, pro3)
            logits = ops.concat((logits, nega.reshape(-1, 1)), axis=1)
            
        logits /= temperature
        
        # Labels for contrastive loss are all zeros (positive sample is at index 0)
        # logits is (N, 1 + num_prev)
        # labels should be (N,) zeros
        labels_con = ops.zeros(logits.shape[0], mindspore.int32)
        
        loss2 = mu * criterion(logits, labels_con)
        loss1 = criterion(out, label)
        loss = loss1 + loss2
        return loss, out

    grad_fn = value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    net.set_train(True)
    
    for epoch in range(epochs):
        epoch_loss_collector = []
        for data, label in train_dataloader.create_tuple_iterator(num_epochs=1):
            label = label.astype(mindspore.int32)
            (loss, _), grads = grad_fn(data, label)
            optimizer(grads)
            epoch_loss_collector.append(loss.asnumpy())
            
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        if logger:
            logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
            
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
    return test_acc

def local_train_net(nets, args, net_dataidx_map, train_dl=None, test_dl=None, global_model = None, prev_model_pool = None, round=None, device="cpu", logger=None):
    acc_list = []
    # global_model.cuda() -> Not needed in MS
    for net_id, net in nets.items():
        # dataidxs = net_dataidx_map[net_id] # Not used here, train_dl is already partitioned? 
        # In utils.py, train_dl is a list of dataloaders?
        # Wait, utils.py get_dataloader returns train_dl, test_dl.
        # In main.py:
        # train_local_dls=[]
        # for net_id, net in nets.items():
        #     dataidxs = net_dataidx_map[net_id]
        #     train_dl_local, _ = get_dataloader(args, dataidxs)
        #     train_local_dls.append(train_dl_local)
        # So train_dl passed here is a list of dataloaders.
        
        train_dl_local = train_dl[net_id]
        n_epoch = args.epochs
        if args.alg == 'fedavg' or args.alg == 'fednova':
            testacc = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args,
                                        device=device, logger=logger)
        elif args.alg == 'fedprox':
            testacc = train_net_fedprox(net_id, net, global_model, train_dl_local, test_dl, n_epoch, args.lr,
                                                  args.optimizer, args.mu, args, device=device, logger=logger)
        elif args.alg == 'moon':
            prev_models=[]
            for i in range(len(prev_model_pool)):
                prev_models.append(prev_model_pool[i][net_id])
            testacc = train_net_fedcon(net_id, net, global_model, prev_models, train_dl_local, test_dl, n_epoch, args.lr,
                                                  args.optimizer, args.mu, args.temperature, args, round, device=device, logger=logger)
        print("Training network %s, n_training: %d, final test acc %f." % (str(net_id), len(net_dataidx_map[net_id]), testacc))
        acc_list.append(testacc)
    
    return acc_list

def local_train_net_scaffold(nets, global_model, c_nets, c_global, args, net_dataidx_map, train_dl=None, test_dl = None, device="cpu", logger=None):
    # Initialize total_delta
    # total_delta = copy.deepcopy(global_model.state_dict())
    # In MS, state_dict returns dict of params.
    # We want to sum deltas.
    
    total_delta = {}
    for param in global_model.trainable_params():
        total_delta[param.name] = ops.zeros_like(param)

    for idx, (net_id, net) in enumerate(nets.items()):
        train_dl_local = train_dl[net_id]
        n_epoch = args.epochs

        testacc, c_delta_para = train_net_scaffold(net_id, net, global_model, c_nets[net_id], c_global, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args, device=device, logger=logger)

        # c_nets[net_id].to('cpu') -> Not needed
        
        for key in total_delta:
            # key is param name. c_delta_para should have same keys.
            if key in c_delta_para:
                total_delta[key] += c_delta_para[key] / len(nets)
        
        print("Training network %s, n_training: %d, final test acc %f." % (str(net_id), len(net_dataidx_map[net_id]), testacc))

    # Update c_global
    for param in c_global.trainable_params():
        if param.name in total_delta:
            param.set_data(param + total_delta[param.name])

    nets_list = list(nets.values())
    return nets_list