import copy
import torch
import numpy as np
import torch.optim as optim
from utils.model import *
from utils.utils import *
from disco import *


def feddyn_alg(args, n_comm_rounds, global_model, party_list_rounds, net_dataidx_map, train_local_dls, test_dl, traindata_cls_counts, device, global_dist, logger):
    best_test_acc=0
    record_test_acc_list = []

    alpha_coef = 1e-2
    n_par = len(get_mdl_params([global_model])[0])  
    local_param_list = np.zeros((args.n_parties, n_par)).astype('float32')  
    init_par_list=get_mdl_params([global_model], n_par)[0]
    clnt_params_list  = np.ones(args.n_parties).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)
    avg_model = copy.deepcopy(global_model)
    all_model = copy.deepcopy(global_model)
    cld_model = copy.deepcopy(global_model)
    cld_mdl_param = get_mdl_params([cld_model], n_par)[0]
    weight_list = np.asarray([len(net_dataidx_map[i]) for i in range(args.n_parties)])
    weight_list = weight_list / np.sum(weight_list) * args.n_parties
    for round in range(n_comm_rounds):
        logger.info("in comm round:" + str(round))
        print("In communication round:" + str(round))
        party_list_this_round = party_list_rounds[round]
        if args.sample_fraction<1.0:
            print(f'Clients this round : {party_list_this_round}')
        cld_mdl_param_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32).cuda()

        for clnt in party_list_this_round:
            train_dataloader=train_local_dls[clnt]
            model = copy.deepcopy(global_model).cuda()
            # Warm start from current avg model
            model.load_state_dict(cld_model.state_dict())
            for params in model.parameters():
                params.requires_grad = True
            # Scale down
            alpha_coef_adpt = alpha_coef / weight_list[clnt] # adaptive alpha coef
            local_param_list_curr = torch.tensor(local_param_list[clnt], dtype=torch.float32, device='cuda')
            loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=alpha_coef+args.reg)
            model.train()
            model.cuda()
            for e in range(args.epochs):
                # Training
                for batch_idx, (batch_x, batch_y) in enumerate(train_dataloader):
                    batch_x = batch_x.cuda()
                    batch_y = batch_y.cuda()
                    
                    batch_x.requires_grad=False
                    batch_y.requires_grad=False
                    
                    optimizer.zero_grad()
                    _,_,y_pred = model(batch_x)
                    
                    ## Get f_i estimate 
                    loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
                    loss_f_i = loss_f_i / list(batch_y.size())[0]
                    
                    # Get linear penalty on the current parameter estimates
                    local_par_list = None
                    for param in model.parameters():
                        if not isinstance(local_par_list, torch.Tensor):
                        # Initially nothing to concatenate
                            local_par_list = param.reshape(-1)
                        else:
                            local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
                    loss_algo = alpha_coef_adpt * torch.sum(local_par_list * (-cld_mdl_param_tensor + local_param_list_curr))
                    loss = loss_f_i + loss_algo
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10) # Clip gradients
                    optimizer.step()
            # Freeze model
            for params in model.parameters():
                params.requires_grad = False
            model.eval()
            curr_model_par = get_mdl_params([model], n_par)[0]
            # No need to scale up hist terms. They are -\nabla/alpha and alpha is already scaled.
            local_param_list[clnt] += curr_model_par-cld_mdl_param
            clnt_params_list[clnt] = curr_model_par
            model.cuda()
            test_acc, conf_matrix, _ = compute_accuracy(model, test_dl, get_confusion_matrix=True, device=device)
            print("Training network %s, n_training: %d, final test acc %f." % (str(clnt), len(train_dataloader.dataset), test_acc))
            model.to('cpu')

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
            avg_mdl_param = np.dot(np.array(fed_avg_freqs).reshape(1,-1), clnt_params_list[party_list_this_round])
            avg_mdl_param = np.squeeze(avg_mdl_param)
        else: # Model aggregation
            avg_mdl_param = np.mean(clnt_params_list[party_list_this_round], axis = 0)
        cld_mdl_param = avg_mdl_param + np.mean(local_param_list, axis=0)
        avg_model = set_client_from_params(copy.deepcopy(global_model), avg_mdl_param)
        all_model = set_client_from_params(copy.deepcopy(global_model), np.mean(clnt_params_list, axis = 0))
        cld_model = set_client_from_params(copy.deepcopy(global_model), cld_mdl_param) 
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
        mkdirs(args.modeldir+'feddyn/')
        if args.save_model:
            torch.save(global_model.state_dict(), args.modeldir+'feddyn/'+'globalmodel'+args.log_file_name+'.pth')
    return record_test_acc_list, best_test_acc