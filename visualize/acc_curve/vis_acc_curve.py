import numpy as np
import matplotlib.pyplot as plt

color_bank = ['blue', 'yellow', 'skyblue', 'green', 'purple']


# fedavg_npy = np.load('/GPFS/data/ruiye/fssl/FedDisco/logs/acc_list/2022-05-23-1642-06.npy')             # aug
# feddisco_part_npy = np.load('/GPFS/data/ruiye/fssl/FedDisco/logs/acc_list/2022-05-23-1712-01.npy')      # aug
# feddisco_aggr_npy_1 = np.load('/GPFS/data/ruiye/fssl/FedDisco/logs/acc_list/2022-05-23-1632-05.npy')    # aug
# fedavg_npy = np.load('/GPFS/data/ruiye/fssl/FedDisco/logs/acc_list/2022-05-23-1545-47.npy')             # no aug
# feddisco_part_npy = np.load('/GPFS/data/ruiye/fssl/FedDisco/logs/acc_list/2022-05-23-1542-50.npy')      # no aug
# feddisco_aggr_npy_1 = np.load('/GPFS/data/ruiye/fssl/FedDisco/logs/acc_list/2022-05-23-1546-11.npy')    # no aug
# feddisco_npy_1 = np.load('/GPFS/data/ruiye/fssl/FedDisco/logs/acc_list/2022-05-16-1608-28.npy')     # a = 0.1, b = 0.05
# feddisco_npy_2 = np.load('/GPFS/data/ruiye/fssl/FedDisco/logs/acc_list/2022-05-16-1547-23.npy')     # a = 0.1, b = 0.05
# feddisco_npy_3 = np.load('/GPFS/data/ruiye/fssl/FedDisco/logs/acc_list/2022-05-16-1549-13.npy')     # a = 0.1, b = 0.1

# # cifar10 / Non-iid 6. 2,3,4,5,10 each has 10 clients. sample rate =0.2
# fedavg_npy = np.load('/GPFS/data/ruiye/fssl/FedDisco/logs/cifar10/acc_list/2022-06-08-0138-48.npy')
# disco_acc_npy_dict = {
#     'disco_a=0.1':'/GPFS/data/ruiye/fssl/FedDisco/logs/cifar10/acc_list/2022-06-08-0145-55.npy',
#     'disco_a=0.4':'/GPFS/data/ruiye/fssl/FedDisco/logs/cifar10/acc_list/2022-06-08-0158-35.npy',
#     'disco_a=0.5':'/GPFS/data/ruiye/fssl/FedDisco/logs/cifar10/acc_list/2022-06-08-0150-51.npy',
#     'disco_a=0.6':'/GPFS/data/ruiye/fssl/FedDisco/logs/cifar10/acc_list/2022-06-08-0156-17.npy'
# }

# # cifar10 / Non-iid 6. 2 has 40 clients, 10 has 10 clients. sample rate =0.2
# fedavg_npy = np.load('/GPFS/data/ruiye/fssl/FedDisco/logs/cifar10/acc_list/2022-06-08-1525-56.npy')
# disco_acc_npy_dict = {
#     'disco_a=0.1':'/GPFS/data/ruiye/fssl/FedDisco/logs/cifar10/acc_list/2022-06-08-1531-53.npy',
#     'disco_a=0.2':'/GPFS/data/ruiye/fssl/FedDisco/logs/cifar10/acc_list/2022-06-08-1532-13.npy',
#     'disco_a=0.3':'/GPFS/data/ruiye/fssl/FedDisco/logs/cifar10/acc_list/2022-06-08-1540-09.npy',
#     'disco_a=0.4':'/GPFS/data/ruiye/fssl/FedDisco/logs/cifar10/acc_list/2022-06-08-2040-16.npy',
#     'disco_a=0.5':'/GPFS/data/ruiye/fssl/FedDisco/logs/cifar10/acc_list/2022-06-08-1540-36.npy',
#     'disco_a=0.6':'/GPFS/data/ruiye/fssl/FedDisco/logs/cifar10/acc_list/2022-06-08-2040-53.npy',
#     'disco_a=0.7':'/GPFS/data/ruiye/fssl/FedDisco/logs/cifar10/acc_list/2022-06-08-1540-49.npy',
#     'disco_a=0.9':'/GPFS/data/ruiye/fssl/FedDisco/logs/cifar10/acc_list/2022-06-08-1541-34.npy'
# }

# # cifar10 / Non-iid 6. 2 has 40 clients, 10 has 5 clients. sample rate =0.2
# fedavg_npy = np.load('/GPFS/data/ruiye/fssl/FedDisco/logs/cifar10/acc_list/2022-06-08-2249-06.npy')
# disco_acc_npy_dict = {
#     'disco_a=0.1':'/GPFS/data/ruiye/fssl/FedDisco/logs/cifar10/acc_list/2022-06-08-2253-18.npy',
#     'disco_a=0.3':'/GPFS/data/ruiye/fssl/FedDisco/logs/cifar10/acc_list/2022-06-08-2253-56.npy',
#     'disco_a=0.5':'/GPFS/data/ruiye/fssl/FedDisco/logs/cifar10/acc_list/2022-06-08-2254-13.npy',
#     'disco_a=0.7':'/GPFS/data/ruiye/fssl/FedDisco/logs/cifar10/acc_list/2022-06-08-2254-21.npy',
#     'disco_a=0.9':'/GPFS/data/ruiye/fssl/FedDisco/logs/cifar10/acc_list/2022-06-08-2254-29.npy'
# }

# # fmnist / Non-iid 6. 2 has 50 clients, 10 has 10 clients. sample rate =0.15
# fedavg_npy = np.load('/GPFS/data/ruiye/fssl/FedDisco/logs/fmnist/acc_list/2022-06-08-2303-34.npy')
# disco_acc_npy_dict = {
#     'disco_a=0.1':'/GPFS/data/ruiye/fssl/FedDisco/logs/fmnist/acc_list/2022-06-08-2306-45.npy',
#     'disco_a=0.3':'/GPFS/data/ruiye/fssl/FedDisco/logs/fmnist/acc_list/2022-06-08-2307-22.npy',
#     'disco_a=0.5':'/GPFS/data/ruiye/fssl/FedDisco/logs/fmnist/acc_list/2022-06-08-2307-47.npy',
#     'disco_a=0.7':'/GPFS/data/ruiye/fssl/FedDisco/logs/fmnist/acc_list/2022-06-08-2308-10.npy',
#     'disco_a=0.9':'/GPFS/data/ruiye/fssl/FedDisco/logs/fmnist/acc_list/2022-06-08-2308-33.npy'
# }

# cifar10 / Dirichlet beta=0.5. K=50. sample rate =0.2
fedavg_npy = np.load('/GPFS/data/ruiye/fssl/FedDisco/logs/cifar10/acc_list/2022-06-09-0039-56.npy')
disco_acc_npy_dict = {
    'disco_a=0.1':'/GPFS/data/ruiye/fssl/FedDisco/logs/cifar10/acc_list/2022-06-09-0040-29.npy',
    'disco_a=0.3':'/GPFS/data/ruiye/fssl/FedDisco/logs/cifar10/acc_list/2022-06-09-0040-42.npy',
    'disco_a=0.5':'/GPFS/data/ruiye/fssl/FedDisco/logs/cifar10/acc_list/2022-06-09-0041-00.npy',
    'disco_a=0.7':'/GPFS/data/ruiye/fssl/FedDisco/logs/cifar10/acc_list/2022-06-09-0041-28.npy',
    'disco_a=0.9':'/GPFS/data/ruiye/fssl/FedDisco/logs/cifar10/acc_list/2022-06-09-0041-38.npy'
}




print(fedavg_npy.shape)

ax = plt.axes()

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Background color
ax.set_facecolor('#eaeaf1')
plt.grid(color='white', linestyle='-', linewidth=1)
ax.set_axisbelow(True)

start_round = 0
end_round = 100
seq = range(len(fedavg_npy))
plt.plot(seq[start_round:end_round], fedavg_npy[start_round:end_round], label='FedAvg', c='k')
plt.plot(seq[start_round:end_round], np.load(disco_acc_npy_dict['disco_a=0.1'])[start_round:end_round], label='FedDisco_Aggregation', c='blue')

plt.legend()
plt.xlabel('Communication Round')
plt.ylabel('Accuracy')
plt.title('cifar10, beta0.5, K=50, partial participation: 0.2')
plt.savefig('20220609_cifar10_noniid1_K50.png')

print(f'FedAvg : {np.mean(fedavg_npy[-5:])}')
fedavg_mask = fedavg_npy>=0.6
print(np.nonzero(fedavg_mask)[0][0])
for i,key in enumerate(disco_acc_npy_dict.keys()):
    feddisco_npy = np.load(disco_acc_npy_dict[key])
    print(f'FedDisco-{key} : {np.mean(feddisco_npy[-5:])}')
    feddisco_mask = feddisco_npy>=0.6
    print(np.nonzero(feddisco_mask)[0][0])