import copy
import mindspore
import mindspore.ops as ops
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

        # global_w = global_model.state_dict()
        # In MS, we can get a dict of params, but better to just keep reference to global_model params
        # or copy them.
        # To broadcast to local nets:
        global_params = {p.name: p.data.asnumpy() for p in global_model.trainable_params()}
        
        if args.server_momentum:
            old_w = {p.name: p.data.asnumpy() for p in global_model.trainable_params()}

        nets_this_round = {k: nets[k] for k in party_list_this_round}
        for net in nets_this_round.values():
            # net.load_state_dict(global_w)
            # Load global params into net
            for p in net.trainable_params():
                if p.name in global_params:
                    p.set_data(mindspore.Tensor(global_params[p.name], p.dtype))

        # Local update
        local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_local_dls, test_dl=test_dl, device=device, logger=logger)
        
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
        # We need to sum up parameters
        # Initialize global_w with zeros
        new_global_w = {name: np.zeros_like(val) for name, val in global_params.items()}
        
        for net_id, net in enumerate(nets_this_round.values()):
            net_para = {p.name: p.data.asnumpy() for p in net.trainable_params()}
            weight = fed_avg_freqs[net_id]
            for key in new_global_w:
                if key in net_para:
                    new_global_w[key] += net_para[key] * weight

        if args.server_momentum:
            delta_w = {}
            for key in new_global_w:
                delta_w[key] = old_w[key] - new_global_w[key]
                # moment_v is a dict of numpy arrays (initialized in main.py)
                moment_v[key] = args.server_momentum * moment_v[key] + (1-args.server_momentum) * delta_w[key]
                new_global_w[key] = old_w[key] - moment_v[key]
        
        # Load back to global_model
        for p in global_model.trainable_params():
            if p.name in new_global_w:
                p.set_data(mindspore.Tensor(new_global_w[p.name], p.dtype))

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
            mindspore.save_checkpoint(global_model, args.modeldir+'fedavg/'+'globalmodel'+args.log_file_name+'.ckpt')

        print((time.time() - start_time) / 60)
    return record_test_acc_list, best_test_acc