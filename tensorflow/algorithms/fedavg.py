import copy
import tensorflow as tf
from utils.model import *
from utils.utils import *
from disco import *
from algorithms.client import local_train_net

import time

def fedavg_alg(args, n_comm_rounds, nets, global_model, party_list_rounds, net_dataidx_map, train_local_dls, test_dl, traindata_cls_counts, moment_v, device, global_dist, logger):
    best_test_acc=0
    record_test_acc_list = []
    start_time = time.time()
    for round in range(n_comm_rounds):
        logger.info("in comm round:" + str(round))
        print("In communication round:" + str(round))
        party_list_this_round = party_list_rounds[round]
        if args.sample_fraction<1.0:
            print(f'Clients this round : {party_list_this_round}')

        global_w = global_model.get_weights()
        print("Global model weights:", global_w[1][:10])
        if args.server_momentum:
            old_w = [tf.identity(w) for w in global_w]

        nets_this_round = {k: nets[k] for k in party_list_this_round}
        for net in nets_this_round.values():
            net.set_weights(global_w)
        # Local update
        local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_local_dls, test_dl=test_dl, device=device, logger=logger)
        for net in nets_this_round.values():
            print("weights:", net.get_weights()[1][:10])
        # Aggregation weight calculation
        total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
        fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]
        if round==0 or args.sample_fraction<1.0:
            print(f'Dataset size weight : {fed_avg_freqs}')

        if args.disco: # Discrepancy-aware collaboration
            distribution_difference = get_distribution_difference(traindata_cls_counts, participation_clients=party_list_this_round, metric=args.measure_difference, hypo_distribution=global_dist)
            fed_avg_freqs = disco_weight_adjusting(fed_avg_freqs, distribution_difference, args.disco_a, args.disco_b)
            if round==0 or args.sample_fraction<1.0:
                print(f'Distribution_difference : {distribution_difference}\nDisco Aggregation Weights : {fed_avg_freqs}')

        # Model aggregation
        new_global_w = []
        for net_id, net in enumerate(nets_this_round.values()):
            net_weights = net.get_weights()
            if net_id == 0:
                new_global_w = [w * fed_avg_freqs[net_id] for w in net_weights]
            else:
                for i in range(len(new_global_w)):
                    new_global_w[i] += net_weights[i] * fed_avg_freqs[net_id]

        if args.server_momentum:
            delta_w = [old_w[i] - new_global_w[i] for i in range(len(new_global_w))]
            for i in range(len(moment_v)):
                moment_v[i] = args.server_momentum * moment_v[i] + (1-args.server_momentum) * delta_w[i]
                new_global_w[i] = old_w[i] - moment_v[i]
        global_model.set_weights(new_global_w)

        # Test
        test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)
        record_test_acc_list.append(test_acc)

        if(best_test_acc<test_acc):
            best_test_acc=test_acc
            logger.info('New Best best test acc:%f'% test_acc)
        logger.info('>> Global Model Test accuracy: %f' % test_acc)
        logger.info('>> Global Model Best accuracy: %f' % best_test_acc)
        print('>> Global Model Test accuracy: %f, Best: %f' % (test_acc, best_test_acc))
        
        mkdirs(args.modeldir+'fedavg/')
        if args.save_model:   
            global_model.save_weights(args.modeldir+'fedavg/'+'globalmodel'+args.log_file_name+'.h5')

        print((time.time() - start_time) / 60)
    return record_test_acc_list, best_test_acc