import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
np.random.seed(0)


min_size = 0
min_require_size = 10
K = 10
beta = 0.5
n_parties = 10
num_samples = 50000
y_train = np.ones(num_samples)
for i in range(K):
    y_train[i*(num_samples//K):(i+1)*(num_samples//K)] = i

N = y_train.shape[0]
net_dataidx_map = {}

while min_size < min_require_size:
    idx_batch = [[] for _ in range(n_parties)]
    for k in range(K):
        idx_k = np.where(y_train == k)[0]
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet(np.repeat(beta, n_parties))
        proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
        min_size = min([len(idx_j) for idx_j in idx_batch])

class_dis = torch.zeros((n_parties, K))
for j in range(n_parties):
    for m in range(K):
        class_dis[j,m] = int((np.array(y_train[idx_batch[j]])==m).sum())
print(class_dis.size())
print(class_dis.sum(dim=1))
print(class_dis.sum(dim=0))
dataset_weight = class_dis.sum(dim=1) / class_dis.sum()
dataset_weight = dataset_weight.unsqueeze(-1)
class_dis = torch.div(class_dis, class_dis.sum(dim=1).unsqueeze(-1))
print(dataset_weight)
print(class_dis)
print(torch.mm(class_dis.T,dataset_weight))
print(dataset_weight.size())
print(class_dis.size())
target = torch.ones((K,1))/K

for round in range(1):
    chosen_client = np.random.choice(range(n_parties), 10, replace=False)
    chosen_dis = class_dis[chosen_client]
    chosen_dataset_weight = dataset_weight[chosen_client]
    chosen_dataset_weight = chosen_dataset_weight / chosen_dataset_weight.sum()
    print(f'>>Round {round}: {chosen_dis}')
    print(f'Dataset_weight {chosen_dataset_weight}')
    weights_variable = Variable(chosen_dataset_weight, requires_grad=True)
    print(f'Initial output : {torch.mm(chosen_dis.T,weights_variable)}')
    for iter in range(50):
        loss_criterion = torch.nn.MSELoss()
        output = torch.mm(chosen_dis.T,weights_variable)
        loss = loss_criterion(output, target)
        loss.backward()
        # print(f'Grad: {weights_variable.grad.data}')
        weights_variable.data = weights_variable.data - 1.0 * weights_variable.grad.data
        weights_variable.grad.data.zero_()
        print(f'Loss : {loss} ; Updated : {weights_variable}')
    print(f'Final output : {output}')


        


        