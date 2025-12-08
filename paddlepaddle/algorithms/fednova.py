import torch
from utils.model import *
from utils.utils import *
from disco import *
from algorithms.client import local_train_net


def fednova_alg(args, n_comm_rounds, nets, global_model, party_list_rounds, net_dataidx_map, train_local_dls, test_dl, traindata_cls_counts, device, global_dist, logger):
    best_test_acc=0
    record_test_acc_list = []

    for round in range(n_comm_rounds):
        logger.info("in comm round:" + str(round))
        print("In communication round:" + str(round))
        party_list_this_round = party_list_rounds[round]
        global_w = global_model.state_dict()
        nets_this_round = {k: nets[k] for k in party_list_this_round}
        for net in nets_this_round.values():
            net.load_state_dict(global_w)

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

        client_step = [(len(net_dataidx_map[r]) // args.batch_size) for r in party_list_this_round]
        tao_eff = 0.0
        for j in range(len(fed_avg_freqs)):
            tao_eff += fed_avg_freqs[j] * client_step[j]
        correct_term = 0.0
        for j in range(len(fed_avg_freqs)):
            correct_term += fed_avg_freqs[j] / client_step[j] * tao_eff
        for key in global_w.keys():
            global_w[key] = (1.0 - correct_term)*global_w[key]

        # Model aggregation
        for net_id, net in enumerate(nets_this_round.values()):
            net_para = net.state_dict()
            for key in net_para:
                global_w[key] += net_para[key] * fed_avg_freqs[net_id] / client_step[net_id] * tao_eff
        global_model.load_state_dict(global_w)
        global_model.cuda()

        # Test
        test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)
        record_test_acc_list.append(test_acc)

        if(best_test_acc<test_acc):
            best_test_acc=test_acc
            logger.info('New Best best test acc:%f'% test_acc)
        logger.info('>> Global Model Test accuracy: %f' % test_acc)
        logger.info('>> Global Model Best accuracy: %f' % best_test_acc)
        print('>> Global Model Test accuracy: %f, Best: %f' % (test_acc, best_test_acc))
        mkdirs(args.modeldir+'fednova/')
        global_model.to('cpu')
        if args.save_model:
            torch.save(global_model.state_dict(), args.modeldir+'fednova/'+'globalmodel'+args.log_file_name+'.pth')
    return record_test_acc_list, best_test_acc