from src import *
import numpy as np
import torch


if __name__ == '__main__':

    nca_train_data = np.load('../dirsoild_3.npy', allow_pickle=True)[:, ..., :28, :-1]
    '''
    for i in range(6, 7):
        nca_train_data2 = np.load('dirsoild_' + str(i) + '.npy', allow_pickle=True)[:, ..., :28, :-1]
        nca_train_data = np.concatenate([nca_train_data, nca_train_data2], axis=0)
        del nca_train_data2
    '''
    nca_train_data = torch.from_numpy(nca_train_data)
    nca_train_data = nca_train_data.permute([0, 1, 5, 4, 2, 3]).type(torch.FloatTensor)
    print(f'Input data shape: {nca_train_data.shape};')

    train_dataset = None

    parameters = {
        "lr": [1e-3],
        "step_size": [700],
        "gamma": [0.5],
        "hid_lay_num": [3],
        "kernel_size": [3],
        "neu_num": [384],
        "epoch": [1],
        "echo_step": [40],
        "rand_seed": [3024],
        "speedup_rate": [[1.0]],
        "batch_size": [4],
        "drop": [0.9],
        "tot_t": [10],
        "EA_cla": [[10, 5, 10]],
        #"loss_weight": [[0.1, 0.1, 0.1]],
        #"retrain": [True],
        #"reg_para": [1e-10],
        #"reg_exp": [2.0],
    }

    # define GPU number
    world_size = 3

    # define validation and test datasize
    valid_ratio = 3. / 10
    test_ratio = 2. / 10

    # run foreach setup
    ensemble_runs(parameters, nca_train_data, valid_ratio, test_ratio, train_dataset, world_size)