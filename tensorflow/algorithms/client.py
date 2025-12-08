import copy
import numpy as np
import tensorflow as tf
from tensorflow import keras
from utils.utils import compute_accuracy


def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device="cpu", logger=None):
    logger.info('Training network %s' % str(net_id))

    if args_optimizer == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = keras.optimizers.Adam(learning_rate=lr, weight_decay=args.reg, amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = keras.optimizers.SGD(learning_rate=lr, momentum=0.9, weight_decay=args.reg)

    criterion = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    # @tf.function
    def train_step(x, target):
        with tf.GradientTape() as tape:
            _, _, out = net(x, training=True)
            loss = criterion(target, out)
        gradients = tape.gradient(loss, net.trainable_variables)
        optimizer.apply_gradients(zip(gradients, net.trainable_variables))
        # for g, v in zip(gradients, net.trainable_variables):
        #     print(v.name, tf.reduce_mean(g).numpy(), tf.reduce_max(g).numpy())
        # exit()
        return loss
    
    for epoch in range(epochs): 
        epoch_loss_collector = []    
        for batch_idx, (x, target) in enumerate(train_dataloader):
            if isinstance(x, np.ndarray):
                x = tf.constant(x, dtype=tf.float32)
            if isinstance(target, np.ndarray):
                target = tf.constant(target, dtype=tf.int64)
            loss = train_step(x, target)
            epoch_loss_collector.append(float(loss.numpy()))
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
    logger.info('>> Test accuracy: %f' % test_acc)
    return test_acc

def train_net_scaffold(net_id, net, global_model, c_local, c_global, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device="cpu", logger=None):
    logger.info('Training network %s' % str(net_id))

    if args_optimizer == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = keras.optimizers.Adam(learning_rate=lr, weight_decay=args.reg, amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = keras.optimizers.SGD(learning_rate=lr, momentum=0.9, weight_decay=args.reg)
    criterion = keras.losses.SparseCategoricalCrossentropy()

    cnt = 0
    c_global_weights = c_global.get_weights()
    c_local_weights = c_local.get_weights()

    @tf.function
    def train_step(x, target):
        with tf.GradientTape() as tape:
            _, _, out = net(x, training=True)
            loss = criterion(target, out)
        gradients = tape.gradient(loss, net.trainable_variables)
        optimizer.apply_gradients(zip(gradients, net.trainable_variables))
        return loss

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            if isinstance(x, np.ndarray):
                x = tf.constant(x, dtype=tf.float32)
            if isinstance(target, np.ndarray):
                target = tf.constant(target, dtype=tf.int64)
            loss = train_step(x, target)
            
            # SCAFFOLD correction
            net_weights = net.get_weights()
            new_weights = []
            for i, w in enumerate(net_weights):
                if i < len(c_global_weights) and i < len(c_local_weights):
                    new_w = w - args.lr * (c_global_weights[i] - c_local_weights[i])
                    new_weights.append(new_w)
                else:
                    new_weights.append(w)
            net.set_weights(new_weights)
            cnt += 1
            epoch_loss_collector.append(float(loss.numpy()))

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    c_new_weights = []
    c_delta_weights = []
    global_model_weights = global_model.get_weights()
    net_weights = net.get_weights()
    c_local_weights = c_local.get_weights()
    for i in range(len(net_weights)):
        if i < len(c_global_weights) and i < len(c_local_weights):
            c_new_w = c_local_weights[i] - c_global_weights[i] + (global_model_weights[i] - net_weights[i]) / (cnt * args.lr)
            c_delta_w = c_new_w - c_local_weights[i]
            c_new_weights.append(c_new_w)
            c_delta_weights.append(c_delta_w)
        else:
            c_new_weights.append(c_local_weights[i] if i < len(c_local_weights) else net_weights[i])
            c_delta_weights.append(tf.zeros_like(net_weights[i]))
    c_local.set_weights(c_new_weights)

    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
    logger.info('>> Test accuracy: %f' % test_acc)
    return test_acc, c_delta_weights

def train_net_fedprox(net_id, net, global_net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, args,
                      device="cpu", logger=None):
    logger.info('Training network %s' % str(net_id))
    if args_optimizer == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = keras.optimizers.Adam(learning_rate=lr, weight_decay=args.reg, amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = keras.optimizers.SGD(learning_rate=lr, momentum=0.9, weight_decay=args.reg)

    criterion = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    global_weights = global_net.get_weights()

    @tf.function
    def train_step(x, target):
        with tf.GradientTape() as tape:
            _, _, out = net(x, training=True)
            loss = criterion(target, out)
            # For fedprox
            fed_prox_reg = 0.0
            net_weights = net.get_weights()
            for param_index, param in enumerate(net_weights):
                fed_prox_reg += (mu / 2) * tf.reduce_sum(tf.square(param - global_weights[param_index]))
            loss += fed_prox_reg
        gradients = tape.gradient(loss, net.trainable_variables)
        optimizer.apply_gradients(zip(gradients, net.trainable_variables))
        return loss

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            if isinstance(x, np.ndarray):
                x = tf.constant(x, dtype=tf.float32)
            if isinstance(target, np.ndarray):
                target = tf.constant(target, dtype=tf.int64)
            loss = train_step(x, target)
            epoch_loss_collector.append(float(loss.numpy()))
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
    return test_acc

def train_net_fedcon(net_id, net, global_net, previous_nets, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, temperature, args,
                      round, device="cpu", logger=None):
    logger.info('Training network %s' % str(net_id))

    if args_optimizer == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = keras.optimizers.Adam(learning_rate=lr, weight_decay=args.reg, amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = keras.optimizers.SGD(learning_rate=lr, momentum=0.9, weight_decay=args.reg)

    criterion = keras.losses.SparseCategoricalCrossentropy()

    @tf.function
    def train_step(x, target):
        with tf.GradientTape() as tape:
            _, pro1, out = net(x, training=True)
            _, pro2, _ = global_net(x, training=False)
            
            posi = tf.reduce_sum(pro1 * pro2, axis=-1) / (tf.norm(pro1, axis=-1) * tf.norm(pro2, axis=-1) + 1e-8)
            logits = tf.expand_dims(posi, -1)

            for previous_net in previous_nets:
                _, pro3, _ = previous_net(x, training=False)
                nega = tf.reduce_sum(pro1 * pro3, axis=-1) / (tf.norm(pro1, axis=-1) * tf.norm(pro3, axis=-1) + 1e-8)
                logits = tf.concat([logits, tf.expand_dims(nega, -1)], axis=1)

            logits = logits / temperature
            labels = tf.zeros(x.shape[0], dtype=tf.int64)

            loss2 = mu * criterion(labels, logits)
            loss1 = criterion(target, out)
            loss = loss1 + loss2
        gradients = tape.gradient(loss, net.trainable_variables)
        optimizer.apply_gradients(zip(gradients, net.trainable_variables))
        return loss

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            if isinstance(x, np.ndarray):
                x = tf.constant(x, dtype=tf.float32)
            if isinstance(target, np.ndarray):
                target = tf.constant(target, dtype=tf.int64)
            loss = train_step(x, target)
            epoch_loss_collector.append(float(loss.numpy()))
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
    return test_acc

def local_train_net(nets, args, net_dataidx_map, train_dl=None, test_dl=None, global_model = None, prev_model_pool = None, round=None, device="cpu", logger=None):
    acc_list = []
    for net_id, net in nets.items():
        dataidxs = net_dataidx_map[net_id]
        train_dl_local=train_dl[net_id]   
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
        print("Training network %s, n_training: %d, final test acc %f." % (str(net_id), len(dataidxs), testacc))
        acc_list.append(testacc)
    return acc_list

def local_train_net_scaffold(nets, global_model, c_nets, c_global, args, net_dataidx_map, train_dl=None, test_dl = None, device="cpu", logger=None):
    global_weights = global_model.get_weights()
    total_delta = [tf.zeros_like(w) for w in global_weights]

    for idx, (net_id, net) in enumerate(nets.items()):
        dataidxs = net_dataidx_map[net_id]
        train_dl_local = train_dl[net_id]
        n_epoch = args.epochs

        testacc, c_delta_weights = train_net_scaffold(net_id, net, global_model, c_nets[net_id], c_global, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args, device=device, logger=logger)

        for i in range(len(total_delta)):
            if i < len(c_delta_weights):
                total_delta[i] += c_delta_weights[i] / len(c_nets)
        print("Training network %s, n_training: %d, final test acc %f." % (str(net_id), len(dataidxs), testacc))

    c_global_weights = c_global.get_weights()
    new_c_global_weights = []
    for i in range(len(c_global_weights)):
        if i < len(total_delta):
            new_c_global_weights.append(c_global_weights[i] + total_delta[i])
        else:
            new_c_global_weights.append(c_global_weights[i])
    c_global.set_weights(new_c_global_weights)

    nets_list = list(nets.values())
    return nets_list