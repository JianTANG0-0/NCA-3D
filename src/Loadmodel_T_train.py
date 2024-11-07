import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import re
import time
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kern_size = 3, pad_size = 1,
                 stride = 1, pad_mode = 'zeros', batch_norm = None, downsample = None):
        super(ResidualBlock, self).__init__()
        if batch_norm:
            self.conv1 = nn.Sequential(
                            nn.Conv3d(in_channels, out_channels, kernel_size = kern_size,
                                      stride = stride, padding = pad_size, padding_mode=pad_mode, bias=False),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU())
            self.conv2 = nn.Sequential(
                            nn.Conv3d(out_channels, out_channels, kernel_size = kern_size,
                                      stride = stride, padding = pad_size, padding_mode=pad_mode, bias=False),
                            nn.BatchNorm2d(out_channels))
        else:
            self.conv1 = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=kern_size,
                          stride=stride, padding=pad_size, padding_mode=pad_mode, bias=False),
                nn.ReLU())
            self.conv2 = nn.Sequential(
                nn.Conv3d(out_channels, out_channels, kernel_size=kern_size,
                          stride=stride, padding=pad_size, padding_mode=pad_mode, bias=False))

        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class NCA(torch.nn.Module):
    def __init__(self, parameterization):
        super(NCA, self).__init__()
        self.class_def = parameterization.get("EA_cla", [9, 4, 9])
        self.tot_cla = self.class_def[0] + self.class_def[1] + self.class_def[2]
        self.input_dimension = parameterization.get("hid_dim", 3) + self.tot_cla + 3
        self.neurons = parameterization.get("neu_num", 32)
        self.n_hidden_layers = parameterization.get("hid_lay_num", 3)
        self.output_dimension = self.input_dimension - 2
        self.kern_size = parameterization.get("kernel_size", 3)
        self.pad_size = self.kern_size//2
        self.pad_mode = parameterization.get("padding_mode", 'zeros')
        self.live_mask_size = parameterization.get("livemask_size", 21)


        self.dr = parameterization.get("drop", 0.0)

        self.input_layer = ResidualBlock(self.input_dimension, self.neurons, self.kern_size, 1,
                                     self.pad_size,
                                     self.pad_mode,None, nn.Conv3d(self.input_dimension, self.neurons, kernel_size=1,
                                          bias=False))
        self.hidden_layers = nn.ModuleList(
                [ResidualBlock(self.neurons, self.neurons, self.kern_size, self.pad_size, 1,
                           self.pad_mode, False)
                 for i in range(self.n_hidden_layers)])

        self.output_layer = nn.Conv3d(self.neurons, self.output_dimension, kernel_size=1,
                                          bias=False)

        self.dropout = nn.Dropout(p=self.dr)

        self.bc_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.cn_loss = nn.CrossEntropyLoss(reduction='none')
        self.loss_weight = parameterization.get("loss_weight", [1., 1., 1.])
        self.reg_loss_w = parameterization.get("reg_para", 0.)
        self.reg_loss_exp = parameterization.get("reg_exp", 2.0)
        self.reg_penalty = self.reg_loss_w != 0.

    def forward(self, x):
        solid_mask, x = self.non_liquid_mask(x)

        input_x = x
        live_mask = self.get_living_mask(input_x) * solid_mask

        x = self.input_layer(x)
        for k, l in enumerate(self.hidden_layers):
            x = l(x)

        if self.dr!= 0.0:
            x = self.dropout(x)
        x = self.output_layer(x)

        #class_change
        y = torch.cat(
            [x[:, :self.tot_cla+1, ...], input_x[:, 6:8, ...]*0.0, x[:, self.tot_cla+1:, ...]],
            axis=1)

        output_x = input_x + y * live_mask

        return output_x

    def non_liquid_mask(self, x):
        solid_mask = (x[:,self.tot_cla+1:self.tot_cla+2,...]>0)* (x[:,self.tot_cla+2:self.tot_cla+3,...]>0)
        # set liquid area to all zero values
        x[:,:self.tot_cla+1,...] = x[:,:self.tot_cla+1, ...] * solid_mask
        x[:,self.tot_cla+3:,...] = x[:,self.tot_cla+3:, ...] * solid_mask
        return solid_mask, x

    def initialize_weights(self):
        torch.nn.init.constant_(self.output_layer.weight.data, 0.0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def get_living_mask(self, x):
        alpha = x[:, self.tot_cla:self.tot_cla+1, ...]
        max_pool = torch.nn.MaxPool3d(kernel_size=self.live_mask_size, stride=1, padding=self.live_mask_size//2)
        alpha1 = (max_pool(alpha) > 0.1) * (alpha < 0.99)
        return alpha1

    def cweigt_loss(self, p, t):
        l_weight = torch.sum(t[:, 3, ...] == 1.0, axis=[1, 2, 3])
        l_s = torch.mean(torch.mean(torch.square(p[:, self.tot_cla:self.tot_cla+1, ...] - t[:, 3:4, ...]), axis=[1, 2, 3, 4]) / (
                    l_weight + 1e-12) )

        l_ea1 = torch.mean(torch.mean(self.cn_loss(p[:, :self.class_def[0], ...],
                                                   t[:, 0, ...].to(torch.long)), axis=[1, 2, 3]) / (l_weight+1e-12))

        l_ea2 = torch.mean(torch.mean(self.cn_loss(p[:, self.class_def[0]:self.class_def[0]+self.class_def[1], ...],
                                                   t[:, 1, ...].to(torch.long)), axis=[1, 2, 3]) / (l_weight+1e-12))

        l_ea3 = torch.mean(torch.mean(self.cn_loss(p[:, self.class_def[0]+self.class_def[1]:self.tot_cla, ...],
                                                   t[:, 2, ...].to(torch.long)), axis=[1, 2, 3]) / (l_weight+1e-12))

        l_ea1 = l_ea1 / (self.loss_weight[0] * l_ea1 / l_s).item()
        l_ea2 = l_ea2 / (self.loss_weight[1] * l_ea2 / l_s).item()
        l_ea3 = l_ea3 / (self.loss_weight[2] * l_ea3 / l_s).item()

        l = l_s + l_ea1 + l_ea2 + l_ea3
        if self.reg_penalty:
            #print(self.regularization().item())
            l += self.reg_loss_w * self.regularization()


        return l

    def regularization(self):
            reg_loss = 0.
            for name, param in self.named_parameters():
                if 'weight' in name:
                    reg_loss = reg_loss + torch.norm(param, self.reg_loss_exp)
            return reg_loss

    def pred_angle(self,x):
        x = x.permute(0, 3, 4, 2, 1).cpu().detach().numpy()
        x = np.concatenate([np.clip(np.argmax(x[..., :self.class_def[0]], axis=-1, keepdims=True) / 9., 0.0, 1.0)* 2.0 * np.pi,
                                  np.clip(np.argmax(x[..., self.class_def[0]:self.class_def[1]+self.class_def[0]], axis=-1, keepdims=True) / 4., 0.0, 1.0) * np.pi,
                                  np.clip(np.argmax(x[..., self.class_def[1]+self.class_def[0]:self.tot_cla], axis=-1, keepdims=True) / 9., 0.0, 1.0)* 2.0 * np.pi,
                                  x[..., self.tot_cla:]], axis=-1)
        return x

# load the CRNN Model from file
def load_model(model_file='./Setup_0'):
    model_para = np.load(model_file+'/model_setting.npy',allow_pickle=True).item()
    model_file = model_file + '/model.pkl'
    ca = NCA(model_para)
    ca.load_state_dict(torch.load(model_file, map_location='cpu'))
    return ca


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0.2):
        """
        Args:
            patience (int): How long to wait after last time validation acc improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation acc improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_acc):
        score = val_acc

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def activation(name):
    if name in ['tanh', 'Tanh']:
        return nn.Tanh()
    elif name in ['relu', 'ReLU']:
        return nn.ReLU(inplace=True)
    elif name in ['gelu']:
        return nn.GELU()
    elif name in ['lrelu', 'LReLU']:
        return nn.LeakyReLU(inplace=True)
    elif name in ['sigmoid', 'Sigmoid']:
        return nn.Sigmoid()
    elif name in ['softplus', 'Softplus']:
        return nn.Softplus(beta=4)
    elif name in ['celu', 'CeLU']:
        return nn.CELU()
    elif name in ['mish', 'Mish']:
        return nn.Mish()
    else:
        raise ValueError('Unknow activation function')

