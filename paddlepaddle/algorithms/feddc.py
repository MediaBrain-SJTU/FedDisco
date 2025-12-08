import copy
import torch
import numpy as np
from utils.model import *
from utils.utils import *
from disco import *


def feddc_alg(args, n_comm_rounds, global_model, party_list_rounds, net_dataidx_map, train_local_dls, test_dl, traindata_cls_counts, device, global_dist, logger):
    best_test_acc=0
    record_test_acc_list = []

    alpha_coef = 1e-2
    n_par = len(get_mdl_params([global_model])[0])
    init_par_list=get_mdl_params([global_model], n_par)[0]
    clnt_params_list  = np.ones(args.n_parties).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) 
    parameter_drifts = np.zeros((args.n_parties, n_par)).astype('float32') 
    state_gadient_diffs = np.zeros((args.n_parties+1, n_par)).astype('float32') 
    avg_model = copy.deepcopy(global_model).cuda()
    all_model = copy.deepcopy(global_model).cuda()
    cld_model = copy.deepcopy(global_model).cuda()
    cld_mdl_param = get_mdl_params([cld_model], n_par)[0]
    weight_list = np.asarray([len(net_dataidx_map[i]) for i in range(args.n_parties)])
    weight_list = weight_list / np.sum(weight_list) * args.n_parties

    for round in range(n_comm_rounds):
        logger.info("in comm round:" + str(round))
        print("In communication round:" + str(round))
        party_list_this_round = party_list_rounds[round]
        if args.sample_fraction<1.0:
            print(f'Clients this round : {party_list_this_round}')
        global_mdl = torch.tensor(cld_mdl_param, dtype=torch.float32, device='cuda')
        delta_g_sum = np.zeros(n_par)
        
        for clnt in party_list_this_round:
            train_dataloader=train_local_dls[clnt]
            model = copy.deepcopy(global_model).cuda()
            model.load_state_dict(cld_model.state_dict())
            for params in model.parameters():
                params.requires_grad = True
            local_update_last = state_gadient_diffs[clnt]
            global_update_last = state_gadient_diffs[-1]/weight_list[clnt]
            alpha = alpha_coef / weight_list[clnt] 
            hist_i = torch.tensor(parameter_drifts[clnt], dtype=torch.float32, device='cuda')

            state_update_diff = torch.tensor(-local_update_last+ global_update_last,  dtype=torch.float32, device='cuda')  
            loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.reg)
            model.train()
            model.cuda()

            for e in range(args.epochs):
                for batch_idx, (batch_x, batch_y) in enumerate(train_dataloader):
                    batch_x = batch_x.cuda()
                    batch_y = batch_y.cuda()
                    batch_x.requires_grad=False
                    batch_y.requires_grad=False
                    optimizer.zero_grad()

                    _,_,y_pred = model(batch_x)
                    
                    ## Get f_i estimate 
                    loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())                  
                    loss_f_i = loss_f_i / list(batch_y.size())[0]   # empirical loss term
                    
                    local_parameter = None
                    for param in model.parameters():
                        if not isinstance(local_parameter, torch.Tensor):
                        # Initially nothing to concatenate
                            local_parameter = param.reshape(-1)
                        else:
                            local_parameter = torch.cat((local_parameter, param.reshape(-1)), 0)
                    
                    loss_cp = alpha/2 * torch.sum((local_parameter - (global_mdl - hist_i))*(local_parameter - (global_mdl - hist_i)))  # penalized term
                    loss = loss_f_i + loss_cp
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10) # Clip gradients to prevent exploding
                    optimizer.step()
            # Freeze model
            for params in model.parameters():
                params.requires_grad = False
            model.eval()
            test_acc, conf_matrix, _= compute_accuracy(model, test_dl, get_confusion_matrix=True, device=device)
            print("Training network %s, n_training: %d, final test acc %f." % (str(clnt), len(train_dataloader.dataset), test_acc))
            curr_model_par = get_mdl_params([model], n_par)[0]
            delta_param_curr = curr_model_par-cld_mdl_param
            parameter_drifts[clnt] += delta_param_curr  # update the local drift variable
            n_minibatch = (np.ceil(len(train_dataloader.dataset)/args.batch_size) * args.epochs).astype(np.int64)
            beta = 1/n_minibatch/(args.lr) 
            state_g = local_update_last - global_update_last + beta * (-delta_param_curr) 
            delta_g_cur = (state_g - state_gadient_diffs[clnt])*weight_list[clnt]
            delta_g_sum += delta_g_cur
            state_gadient_diffs[clnt] = state_g
            clnt_params_list[clnt] = curr_model_par

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
            avg_mdl_param_sel = np.dot(np.array(fed_avg_freqs).reshape(1,-1), clnt_params_list[party_list_this_round])
            avg_mdl_param_sel = np.squeeze(avg_mdl_param_sel)  
        else: # Model aggregation          
            avg_mdl_param_sel = np.mean(clnt_params_list[party_list_this_round], axis = 0)
        delta_g_cur = (1 / args.n_parties) * delta_g_sum 
        state_gadient_diffs[-1] += delta_g_cur  
        cld_mdl_param = avg_mdl_param_sel + np.mean(parameter_drifts, axis=0)  
        avg_model = set_client_from_params(copy.deepcopy(global_model), avg_mdl_param_sel)
        all_model = set_client_from_params(copy.deepcopy(global_model), np.mean(clnt_params_list, axis = 0))
        cld_model = set_client_from_params(copy.deepcopy(global_model).cuda(), cld_mdl_param) 
        avg_model.cuda()

        # Test
        test_acc, conf_matrix, _= compute_accuracy(avg_model, test_dl, get_confusion_matrix=True, device=device)
        record_test_acc_list.append(test_acc)
        if(best_test_acc<test_acc):
            best_test_acc=test_acc
            logger.info('New best test acc:%f'% test_acc)
        logger.info('>> Global Model Test accuracy: %f' % test_acc)
        logger.info('>> Global Model Best accuracy: %f' % best_test_acc)
        print('>> Global Model Test accuracy: %f, Best: %f' % (test_acc, best_test_acc))
        avg_model.to('cpu')
        mkdirs(args.modeldir+'feddc/')
        if args.save_model:
            torch.save(global_model.state_dict(), args.modeldir+'feddc/'+'globalmodel'+args.log_file_name+'.pth')
    return record_test_acc_list, best_test_acc