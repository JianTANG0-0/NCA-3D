import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import re
import time
import torch.nn.functional as F


# CRNN models
def get_living_mask(x):
    alpha = x[:, 90:91, ...]#torch.argmax(x[:, 20:22, :, :], axis=1, keepdims=True).to(torch.float32)
    max_pool = torch.nn.MaxPool3d(kernel_size=21, stride=1, padding=21//2)
    alpha1 = (max_pool(alpha) > 0.1) * (alpha < 0.99)
    return alpha1

def non_liquid_mask(x):
    return (x[:,91:92,...]>0)* (x[:,92:93,...]>0)


class NCA(torch.nn.Module):
    def __init__(self, parameterization):
        super(NCA, self).__init__()
        self.input_dimension = parameterization.get("in_dim", 16)+87
        self.neurons = parameterization.get("neu_num", 32)
        self.n_hidden_layers = parameterization.get("hid_lay_num", 3)
        self.output_dimension = self.input_dimension - 3
        self.kern_size = parameterization.get("kernel_size", 3)
        self.pad_size = self.kern_size//2 # parameterization.get("pad_size", 1)
        # self.drop_para_1 = parameterization.get("drop_para",0.0)
        self.pad_mode = parameterization.get("padding_mode", 'zeros')
        # define dilated Conv layers
        self.n_dicon_lay = parameterization.get("dic_lay_num", 0)
        self.dicon_neurons = parameterization.get("dic_neu_num", 0)
        self.dr = parameterization.get("drop", 0.0)
        if self.n_dicon_lay != 0:
            self.dic_input_layer = nn.Conv3d(self.input_dimension, self.dicon_neurons, kernel_size=3,
                                             padding=1+self.kern_size//2, padding_mode=self.pad_mode, dilation=1+self.kern_size//2,
                                             bias=False)
            self.dic_c_con = nn.Conv3d(self.neurons + self.dicon_neurons, self.neurons, kernel_size=self.kern_size, padding=self.pad_size,
                           padding_mode=self.pad_mode,  bias=False)
            if self.n_dicon_lay > 1:
                self.dic_layers = nn.ModuleList(
                    [nn.Conv3d(self.dicon_neurons + self.neurons, self.dicon_neurons, kernel_size=3,
                                             padding=1+self.kern_size//2, padding_mode=self.pad_mode, dilation=1+self.kern_size//2,bias=False)
                     for _ in range(self.n_dicon_lay - 1)])
            else:
                self.dic_layers = nn.ModuleList([])

        self.input_layer = nn.Conv3d(self.input_dimension, self.neurons, kernel_size=self.kern_size,
                                     padding=self.pad_size,
                                     padding_mode=self.pad_mode, bias=False)
        if self.n_dicon_lay > 1:
            self.hid_with_dilay = nn.ModuleList(
                [nn.Conv3d(self.neurons + self.dicon_neurons, self.neurons, kernel_size=self.kern_size, padding=self.pad_size,
                           padding_mode=self.pad_mode, bias=False)
                 for _ in range(self.n_dicon_lay-1)])
            self.hidden_layers = nn.ModuleList(
                [nn.Conv3d(self.neurons, self.neurons, kernel_size=self.kern_size, padding=self.pad_size,
                           padding_mode=self.pad_mode, bias=False)
                 for _ in range(self.n_hidden_layers-self.n_dicon_lay)])
        else:
            self.hid_with_dilay = nn.ModuleList([])
            self.hidden_layers = nn.ModuleList(
                [nn.Conv3d(self.neurons, self.neurons, kernel_size=self.kern_size, padding=self.pad_size, padding_mode=self.pad_mode, bias=False)
                 for i in range(self.n_hidden_layers)])

        if self.n_hidden_layers == self.n_dicon_lay-1:
            self.output_layer = nn.Conv3d(self.neurons+self.dicon_neurons, self.output_dimension, kernel_size=1, bias=False)
        else:
            self.output_layer = nn.Conv3d(self.neurons, self.output_dimension, kernel_size=1,
                                          bias=False)

        # self.output_layer.weight.data.zero_()
        self.activation = torch.nn.ReLU()
        self.dropout = nn.Dropout(p=self.dr)



    def forward(self, x):
        solid_mask = non_liquid_mask(x)

        x[:,:91,...] = x[:,:91, ...] * solid_mask
        x[:,94:,...] = x[:,94:, ...] * solid_mask

        input_x = x
        live_mask = get_living_mask(input_x)

        if self.n_dicon_lay != 0:
            x2 = x
            x2 = self.activation(self.dic_input_layer(x2))
            x = self.activation(self.input_layer(x))
            #print(x.shape,x2.shape)
            x = torch.cat([x, x2], axis=1)
        else:
            x = self.activation(self.input_layer(x))
        #x2 = torch.clone(x)
        if self.n_dicon_lay > 1:
            for k, (l,dl) in enumerate(zip(self.hid_with_dilay, self.dic_layers)):
                x2 = x
                x2 = self.activation(dl(x2))
                x = self.activation(l(x))
                x = torch.cat([x, x2], axis=1)
            x = self.activation(self.dic_c_con(x))
            for j in range(self.n_hidden_layers-self.n_dicon_lay):
                l = self.hidden_layers[j]
                x = self.activation(l(x))
        else:
            if self.n_dicon_lay != 0:
                x = self.activation(self.dic_c_con(x))
            for k, l in enumerate(self.hidden_layers):
                x = x + self.activation(l(x))

        if self.dr!= 0.0:
            x = self.dropout(x)
        y1 = self.output_layer(x)#torch.cat([x, x2], axis=1))

        #class_change
        y = torch.cat(
            [y1[:, :91, ...], input_x[:, 6:9, ...]*0.0, y1[:, 91:, ...]],
            axis=1)

        output_x = input_x + y * live_mask * solid_mask

        return output_x

    def initialize_weights(self):
        torch.nn.init.constant_(self.output_layer.weight.data, 0.0)
  

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    m.bias.data.zero_()

def regularization(model,regularization_exp):
        reg_loss = 0
        for name, param in model.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(param, regularization_exp)
        return reg_loss

def weigt_loss(p, t):
    l = torch.mean(torch.square(p[:, :4, ...] - t[:, :4, ...]), axis=[1, 2, 3, 4])
    l_weight = torch.sum(t[:, 3, ...] == 1.0, axis=[1, 2, 3])
    #overflow_loss = (p-p.clamp(-1.0, 1.0)).abs().sum()
    l = torch.mean(l / l_weight) #+ overflow_loss
    return l

#classification losses:  class_change
bc_loss = nn.BCEWithLogitsLoss(reduction='none')
cn_loss = nn.CrossEntropyLoss(reduction='none')
def cweigt_loss(p, t):
    l_weight = torch.sum(t[:, 3, ...] == 1.0, axis=[1, 2, 3])
    l_s = torch.mean(torch.mean(torch.square(p[:, 90:91, ...] - t[:, 3:4, ...]), axis=[1, 2, 3, 4]) / (
                l_weight + 1e-12) ) # torch.mean(bc_loss(p[:, 20:22, ...], t[:, 3:5, ...].to(torch.float32)), axis=[1, 2, 3])

    l_ea1 = torch.mean(torch.mean(cn_loss(p[:, :36, ...], t[:, 0, ...].to(torch.long)), axis=[1, 2, 3]) / (l_weight+1e-12))
    l_ea1 = l_ea1/(0.1*l_ea1/l_s).item()
    l_ea2 = torch.mean(torch.mean(cn_loss(p[:, 36:54, ...], t[:, 1, ...].to(torch.long)), axis=[1, 2, 3]) / (l_weight+1e-12))
    l_ea2 = l_ea2/(0.1*l_ea2/l_s).item()
    l_ea3 = torch.mean(torch.mean(cn_loss(p[:, 54:90, ...], t[:, 2, ...].to(torch.long)), axis=[1, 2, 3]) / (l_weight+1e-12))
    l_ea3 = l_ea3/(0.1*l_ea3/l_s).item()

    l = l_s + l_ea1 + l_ea2+ l_ea3

    #overflow_loss = (p-p.clamp(-1.0, 1.0)).abs().sum()
    #print(torch.mean(l_s/l_weight).item(),torch.mean(l_ea1/l_weight).item(),torch.mean(l_ea2/l_weight).item(),torch.mean(l_ea3/l_weight).item())
    #l = # + torch.mean(torch.abs(p[:,57:,...]-0.0), axis=[1, 2, 3, 4])/500.)
    return l, l_ea1.item(), l_ea2.item(), l_ea3.item(), l_s.item()

# load the CRNN Model from file
def load_model(model_file='./Setup_0'):
    model_para = np.load(model_file+'/model_setting.npy',allow_pickle=True).item()
    model_file = model_file + '/model.pkl'
    ca = NCA(model_para)
    ca.load_state_dict(torch.load(model_file, map_location='cpu'))
    #ca = torch.load(model_file, map_location='cpu')
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


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation, pad_mod, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, padding_mode=pad_mod, bias=False),
            #nn.BatchNorm3d(mid_channels),
            activation,
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, padding_mode=pad_mod, bias=False),
            #nn.BatchNorm3d(out_channels),
            activation,
        )

    def forward(self, x):
        return self.conv(x)


class Unet(torch.nn.Module):
    def __init__(self, parameterization):
        super(Unet, self).__init__()
        self.in_channels = parameterization.get('in_dim', 8)+35*3
        self.out_channels = self.in_channels - 3
        self.features = parameterization.get("features", 64) * [1, 2, 4, 8]
        self.layn = parameterization.get("hid_lay_num", 4)
        self.features = self.features[:self.layn - 1]
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2)  # , stride=2)
        self.activation = activation(parameterization.get('activation', 'relu'))
        self.pad_mod = parameterization.get("padding_mode", "zeros")
        self.dr = parameterization.get("drop", 0.0)
        # DOWN part of the UNet
        for feature in self.features:
            self.downs.append(DoubleConv(self.in_channels, feature, self.activation, self.pad_mod))
            self.in_channels = feature

        # UP part of the UNet
        for feature in reversed(self.features):
            self.ups.append(
                nn.ConvTranspose3d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature, self.activation, self.pad_mod))
        self.bottleneck = DoubleConv(self.features[-1], self.features[-1] * 2, self.activation, self.pad_mod)
        self.final_conv = nn.Conv3d(self.features[0], self.out_channels, kernel_size=1)

    def forward(self, x):
        input_x = x
        # state filter
        live_mask = get_living_mask(input_x)
        solid_mask = non_liquid_mask(input_x)

        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.size()[2:] != skip_connection.size()[2:]:
                xy, xx, xz = x.size()[3],x.size()[2],x.size()[4]
                sy, sx, sz = skip_connection.size()[3], skip_connection.size()[2], skip_connection.size()[4]
                maxy = max(xy, sy)
                maxx = max(xx, sx)
                maxz = max(xz, sz)

                xdiffy = maxy - xy
                xdiffx = maxx - xx
                xdiffz = maxz - xz
                if xdiffy + xdiffy + xdiffz !=0:
                    x = F.pad(x, [xdiffx // 2, xdiffx - xdiffx // 2, xdiffz // 2, xdiffz - xdiffz // 2,
                              xdiffy // 2, xdiffy - xdiffy // 2])

                sdiffy = maxy - sy
                sdiffx = maxx - sx
                sdiffz = maxz - sz
                if sdiffy + sdiffy + sdiffz != 0:
                    skip_connection = F.pad(skip_connection, [sdiffx // 2, sdiffx - sdiffx // 2, sdiffz // 2, sdiffz - sdiffz // 2,
                              sdiffy // 2, sdiffy - sdiffy // 2])

            print(skip_connection.shape, x.shape)
            concat_skip = torch.cat([skip_connection, x], dim=1)
            x = self.ups[idx + 1](concat_skip)

        if self.dr!= 0.0:
            x = self.dropout(x)

        y1 = self.final_conv(x)

        #class_change
        y = torch.cat(
            [y1[:, :109, ...], input_x[:, 6:9, ...] * 0.0, y1[:, 109:, ...]],
            axis=1)

        output_x = input_x + y * live_mask * solid_mask * (input_x[:,109:110,...]>0)* (input_x[:,110:111,...]>0)

        return output_x

    def initialize_weights(self):
        torch.nn.init.constant_(self.final_conv.weight.data, 0.0)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                if m.bias is not None:
                    m.bias.data.zero_()


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

