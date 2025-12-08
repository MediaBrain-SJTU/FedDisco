import numpy as np
from scipy import special


def get_distribution_difference(client_cls_counts, participation_clients, metric, hypo_distribution):
    local_distributions = client_cls_counts[np.array(participation_clients),:]
    local_distributions = local_distributions / local_distributions.sum(axis=1)[:,np.newaxis]
    
    if metric=='cosine':
        similarity_scores = local_distributions.dot(hypo_distribution)/ (np.linalg.norm(local_distributions, axis=1) * np.linalg.norm(hypo_distribution))
        difference = 1.0 - similarity_scores
    elif metric=='only_iid':
        similarity_scores = local_distributions.dot(hypo_distribution)/ (np.linalg.norm(local_distributions, axis=1) * np.linalg.norm(hypo_distribution))
        difference = np.where(similarity_scores>0.9, 0.01, float('inf'))
    elif metric=='l1':
        difference = np.linalg.norm(local_distributions-hypo_distribution, ord=1, axis=1)
    elif metric=='l2':
        difference = np.linalg.norm(local_distributions-hypo_distribution, axis=1)
    elif metric=='kl':
        difference = special.kl_div(local_distributions, hypo_distribution)
        difference = np.sum(difference, axis=1)

        difference = np.array([0 for _ in range(len(difference))]) if np.sum(difference) == 0 else difference / np.sum(difference)
    return difference

def disco_weight_adjusting(old_weight, distribution_difference, a, b):
    weight_tmp = old_weight - a*distribution_difference + b

    if np.sum(weight_tmp>0)>0:
        new_weight = np.copy(weight_tmp)
        new_weight[new_weight<0.0]=0.0

    total_normalizer = sum([new_weight[r] for r in range(len(old_weight))])
    new_weight = [new_weight[r] / total_normalizer for r in range(len(old_weight))]
    return new_weight
