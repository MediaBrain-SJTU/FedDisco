import numpy as np
import sys
from scipy import special


def get_distribution_difference(client_cls_counts, participation_clients, metric):
    global_distribution = np.ones(client_cls_counts.shape[1])/client_cls_counts.shape[1]
    local_distributions = client_cls_counts[np.array(participation_clients),:]
    local_distributions = local_distributions / local_distributions.sum(axis=1)[:,np.newaxis]
    
    if metric=='cosine':
        similarity_scores = local_distributions.dot(global_distribution)/ (np.linalg.norm(local_distributions, axis=1) * np.linalg.norm(global_distribution))
        difference = 1.0 - similarity_scores
    elif metric=='only_iid':
        similarity_scores = local_distributions.dot(global_distribution)/ (np.linalg.norm(local_distributions, axis=1) * np.linalg.norm(global_distribution))
        difference = np.where(similarity_scores>0.9, 0.01, float('inf'))
    elif metric=='l1':
        difference = np.linalg.norm(local_distributions-global_distribution, ord=1, axis=1)
    elif metric=='l2':
        difference = np.linalg.norm(local_distributions-global_distribution, axis=1)
    elif metric=='kl':
        difference = special.kl_div(local_distributions, global_distribution)
        difference = np.sum(difference, axis=1)
    return difference
