import torch
from torch.utils.data import Dataset
import os
import numpy as np
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from torch.utils.data import DataLoader


class DefDataset(Dataset):
    def __init__(self, parameterization, data, shuffle=None):
        self.class_def = parameterization.get("EA_cla", [9, 4, 9])
        self.tot_cla = self.class_def[0] + self.class_def[1] + self.class_def[2]
        self.channel_num = parameterization.get("hid_dim", 3) + 6
        self.ini_t = parameterization.get("initial_frame", [0])
        self.num_t = parameterization.get("tot_t", 18)
        self.sp_rate_l = parameterization.get("delat_t", [1])
        self.x, self.y = self.red_t(data)
        self.shuffle = shuffle

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def red_t(self,  data):
        for k in range(len(self.sp_rate_l)):
            sp_rate = int(self.sp_rate_l[k])
            for i in range(len(self.ini_t)):
                if (i == 0) & (k == 0):
                    x_initial = np.concatenate(
                        [data[:, self.ini_t[i], ...].numpy(), (data[:, self.ini_t[i] + sp_rate, ...].numpy())[:, -1:, ...]], axis=1)
                    y = data[:, (self.ini_t[i] + sp_rate):data.shape[1]:sp_rate].numpy()
                    if y.shape[1] > self.num_t:
                        y = y[:, :self.num_t]
                    elif y.shape[1] < self.num_t:
                        while y.shape[1] < self.num_t:
                            y = np.concatenate((y, y[:, -1:]), axis=1)
                else:
                    x_initial = np.concatenate([x_initial,
                                                np.concatenate([data[:, self.ini_t[i], ...].numpy(),
                                                                (data[:, self.ini_t[i] + sp_rate, ...].numpy())[:, -1:, ...]], axis=1)], axis=0)
                    y_tmp = data[:, (self.ini_t[i] + sp_rate):data.shape[1]:sp_rate].numpy()
                    if y_tmp.shape[1] > self.num_t:
                        y_tmp = y_tmp[:, :self.num_t]
                    elif y_tmp.shape[1] < self.num_t:
                        while y_tmp.shape[1] < self.num_t:
                            y_tmp = np.concatenate((y_tmp, y_tmp[:, -1:]), axis=1)
                    y = np.concatenate((y, y_tmp), axis=0)

        seed = np.zeros([x_initial.shape[0], self.channel_num, x_initial.shape[2], x_initial.shape[3], x_initial.shape[4]],
                        np.float32)
        seed[:, :x_initial.shape[1], ...] = x_initial.astype(np.float32)
        x = torch.from_numpy(seed)
        y = torch.from_numpy(y)

        # split train/valid set
        if None:
            ord = np.array(range(x.shape[0]))
            np.random.shuffle(ord)
            # print(ord)
            x = x[ord]
            y = y[ord]

        # one hot encoding: class_change
        x = torch.cat([F.one_hot(x[:, 0, ...].to(torch.int64), self.class_def[0]).permute(0, 4, 1, 2, 3),
                       F.one_hot(x[:, 1, ...].to(torch.int64), self.class_def[1]).permute(0, 4, 1, 2, 3),
                       F.one_hot(x[:, 2, ...].to(torch.int64), self.class_def[2]).permute(0, 4, 1, 2, 3),
                       x[:, 3:, ...]], axis=1)

        return x, y


def prepare(dataset, rank, world_size, batch_size=20, pin_memory=False, num_workers=0):
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory,
                                num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size)

    return dataloader