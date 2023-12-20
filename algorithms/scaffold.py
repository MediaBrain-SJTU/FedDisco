import torch
from utils.model import *
from utils.utils import *
from disco import *
from algorithms.client import local_train_net_scaffold


def scaffold_alg(args, nets, global_model, party_list_rounds, net_dataidx_map, train_local_dls, test_dl, traindata_cls_counts, device, global_dist, logger):    
    best_test_acc=0
    record_test_acc_list = []

    c_nets = init_nets(args.n_parties, args, device='cpu')
    c_globals = init_nets(1, args, device='cpu')
    c_global = c_globals[0]
    c_global_para = c_global.state_dict()
    for net_id, net in c_nets.items():
        net.load_state_dict(c_global_para)

    for round in range(args.comm_round):
        logger.info("in comm round:" + str(round))
        print("In communication round:" + str(round))

        party_list_this_round = party_list_rounds[round]
        if args.sample_fraction<1.0:
            print(f'Clients this round : {party_list_this_round}')

        global_para = global_model.state_dict()
        nets_this_round = {k: nets[k] for k in party_list_this_round}
        for net in nets_this_round.values():
            net.load_state_dict(global_para)

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
        
        # Local update
        local_train_net_scaffold(nets_this_round, global_model, c_nets, c_global, args, net_dataidx_map, train_dl=train_local_dls, test_dl=test_dl, device=device, logger=logger)

        # Model aggregation
        for idx in range(len(party_list_this_round)):
            net_para = nets[party_list_this_round[idx]].cpu().state_dict()
            if idx == 0:
                for key in net_para:
                    global_para[key] = net_para[key] * fed_avg_freqs[idx]
            else:
                for key in net_para:
                    global_para[key] += net_para[key] * fed_avg_freqs[idx]
        global_model.load_state_dict(global_para)
        global_model.cuda()

        # Test            
        test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)
        record_test_acc_list.append(test_acc)
        global_model.to('cpu')
        if(best_test_acc<test_acc):
            best_test_acc=test_acc
            logger.info('New Best best test acc:%f'% test_acc)
        logger.info('>> Global Model Test accuracy: %f' % test_acc)
        logger.info('>> Global Model Best accuracy: %f' % best_test_acc)
        print('>> Global Model Test accuracy: %f, Best: %f' % (test_acc, best_test_acc))
        mkdirs(args.modeldir+'scaffold/')
        
        if args.save_model:
            torch.save(global_model.state_dict(), args.modeldir+'scaffold/'+'globalmodel'+args.log_file_name+'.pth')
    return record_test_acc_list, best_test_acc