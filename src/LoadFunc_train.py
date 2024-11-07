import os
import io
import PIL.Image, PIL.ImageDraw
import base64
import zipfile
import json
import glob
import itertools
import random
from IPython.display import Image, HTML, clear_output
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import math
import time
import multiprocessing
import argparse
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist

from .Loadmodel_T_train import *
from torch.nn import functional as F
import matplotlib.pyplot as plt
from torch.nn.parallel import DistributedDataParallel as DDP
from .Dataset import DefDataset, prepare
import wandb
import os
import logging


os.environ['FFMPEG_BINARY'] = 'ffmpeg'
import moviepy.editor as mvp
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter


#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['FFMPEG_BINARY'] = 'ffmpeg'


def zoom(img, scale=4):
  img = np.repeat(img, scale, 0)
  img = np.repeat(img, scale, 1)
  return img

class VideoWriter:
  def __init__(self, filename, fps=1.0, **kw):
    self.writer = None
    self.params = dict(filename=filename, fps=fps, **kw)

  def add(self, img):
    img = np.asarray(img)
    if self.writer is None:
      h, w = img.shape[:2]
      self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
    if img.dtype in [np.float32, np.float64]:
      img = np.uint8(img.clip(0, 1)*255)
    if len(img.shape) == 2:
      img = np.repeat(img[..., None], 3, -1)
    self.writer.write_frame(img)

  def close(self):
    if self.writer:
      self.writer.close()

  def __enter__(self):
    return self

  def __exit__(self, *kw):
    self.close()



def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def setup_logging(log_file_path):
    """Set up logging to file and console."""
    # Create a custom logger
    logger = logging.getLogger("NCA_Logger")
    logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all messages

    # Remove existing handlers to prevent duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a file handler to write logs to a file
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)

    # Create a console handler to output logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Create a formatter and set it for both handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Return the configured logger
    return logger


def close_logging_handlers():
    """Flush and close all logging handlers to ensure logs are saved."""
    handlers = logging.root.handlers[:]
    for handler in handlers:
        handler.flush()
        handler.close()
        logging.root.removeHandler(handler)

def cal_misori(pred, true):
    '''
    calculate the misorientation angle between two given groups of Euler angles
    :param pred: 2D plane of pixel Euler angles fron NCA prediction
    :param true: 2D plane of pixel Euler angles fron CA truth
    :return: misorientation angle map
    '''
    p1 = pred[:, :, 0]
    p = pred[:, :, 1]
    p2 = pred[:, :, 2]
    q1 = true[:, :, 0]
    q = true[:, :, 1]
    q2 = true[:, :, 2]

    nx = p.shape[0]
    ny = p.shape[1]

    t1 = np.zeros((nx, ny, 24))
    t2 = np.zeros((nx, ny, 24))
    t3 = np.zeros((nx, ny, 24))
    theta = np.zeros((nx, ny, 24))
    g1 = np.zeros((nx, ny, 3, 3))
    g2 = np.zeros((nx, ny, 3, 3))
    gp = np.zeros((nx, ny, 3, 3))
    gp1 = np.zeros((nx, ny, 3, 3))
    gp2 = np.zeros((nx, ny, 3, 3))
    gq = np.zeros((nx, ny, 3, 3))
    gq1 = np.zeros((nx, ny, 3, 3))
    gq2 = np.zeros((nx, ny, 3, 3))
    m = np.zeros((nx, ny, 24, 3, 3))

    # converting in the form of matrices for both grains
    gp1[:, :, 0, 0] = np.cos(p1)
    gp1[:, :, 1, 0] = -np.sin(p1)
    gp1[:, :, 0, 1] = np.sin(p1)
    gp1[:, :, 1, 1] = np.cos(p1)
    gp1[:, :, 2, 2] = 1
    gp2[:, :, 0, 0] = np.cos(p2)
    gp2[:, :, 1, 0] = -np.sin(p2)
    gp2[:, :, 0, 1] = np.sin(p2)
    gp2[:, :, 1, 1] = np.cos(p2)
    gp2[:, :, 2, 2] = 1
    gp[:, :, 0, 0] = 1
    gp[:, :, 1, 1] = np.cos(p)
    gp[:, :, 1, 2] = np.sin(p)
    gp[:, :, 2, 1] = -np.sin(p)
    gp[:, :, 2, 2] = np.cos(p)
    gq1[:, :, 0, 0] = np.cos(q1)
    gq1[:, :, 1, 0] = -np.sin(q1)
    gq1[:, :, 0, 1] = np.sin(q1)
    gq1[:, :, 1, 1] = np.cos(q1)
    gq1[:, :, 2, 2] = 1
    gq2[:, :, 0, 0] = np.cos(q2)
    gq2[:, :, 1, 0] = -np.sin(q2)
    gq2[:, :, 0, 1] = np.sin(q2)
    gq2[:, :, 1, 1] = np.cos(q2)
    gq2[:, :, 2, 2] = 1
    gq[:, :, 0, 0] = 1
    gq[:, :, 1, 1] = np.cos(q)
    gq[:, :, 1, 2] = np.sin(q)
    gq[:, :, 2, 1] = -np.sin(q)
    gq[:, :, 2, 2] = np.cos(q)
    g1 = np.matmul(np.matmul(gp2, gp), gp1)
    g2 = np.matmul(np.matmul(gq2, gq), gq1)

    # symmetry matrices considering the 24 symmteries for cubic system
    T = np.zeros((24, 3, 3));
    T[0, :, :] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    T[1, :, :] = [[0, 0, -1], [0, -1, 0], [-1, 0, 0]]
    T[2, :, :] = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]
    T[3, :, :] = [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]
    T[4, :, :] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    T[5, :, :] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    T[6, :, :] = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
    T[7, :, :] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    T[8, :, :] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    T[9, :, :] = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
    T[10, :, :] = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
    T[11, :, :] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    T[12, :, :] = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    T[13, :, :] = [[0, 0, -1], [-1, 0, 0], [0, 1, 0]]
    T[14, :, :] = [[0, -1, 0], [0, 0, 1], [-1, 0, 0]]
    T[15, :, :] = [[0, 1, 0], [0, 0, -1], [-1, 0, 0]]
    T[16, :, :] = [[0, 0, -1], [1, 0, 0], [0, -1, 0]]
    T[17, :, :] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    T[18, :, :] = [[0, -1, 0], [0, 0, -1], [1, 0, 0]]
    T[19, :, :] = [[0, 1, 0], [1, 0, 0], [0, 0, -1]]
    T[20, :, :] = [[-1, 0, 0], [0, 0, 1], [0, 1, 0]]
    T[21, :, :] = [[0, 0, 1], [0, -1, 0], [1, 0, 0]]
    T[22, :, :] = [[0, -1, 0], [-1, 0, 0], [0, 0, -1]]
    T[23, :, :] = [[-1, 0, 0], [0, 0, -1], [0, -1, 0]]

    T = np.array(T[None, None, ...])

    # finding the 24 misorientation matrices(also can be calculated for 576 matrices)
    for i in range(24):
        m[:, :, i, :, :] = np.matmul(np.linalg.inv(np.matmul(T[:, :, i, :, :], g1)), g2)
        t1[:, :, i] = m[:, :, i, 0, 0]
        t2[:, :, i] = m[:, :, i, 1, 1]
        t3[:, :, i] = m[:, :, i, 2, 2]
        theta[:, :, i] = np.arccos(0.5 * (t1[:, :, i] + t2[:, :, i] + t3[:, :, i] - 1))

    # minimum of 24 angles is taken as miorientation angle
    ansRad = np.nanmin(theta, axis=-1)
    ansTheta = ansRad * 180.0 / np.pi
    return ansTheta


def takesolid(ea_cs):
    solid_pos = np.where(ea_cs[..., 3:4] > 0.99)
    return ea_cs[solid_pos[0], solid_pos[1], :3]


def predict_fig(x, y, ca, epoch, folder_name, note, min_number=1):
    if min_number > len(x):
        min_number = len(x)
    x = ca.pred_angle(x[:min_number])
    target_valid = (y[:min_number]).permute(0, 1, 4, 5, 3, 2).cpu().detach().numpy()
    cmap = plt.get_cmap("turbo")  # define a colormap
    acc = []
    for ea_i in range(min_number):
        sam_i = x[ea_i, ..., :3]
        sam_i_rgb = np.concatenate(
            [sam_i[..., 0:1] / 2.0 / np.pi, sam_i[..., 1:2] / np.pi, sam_i[..., 2:3] / 2.0 / np.pi], axis=-1)
        target_i = target_valid[ea_i, -1, ..., :3]
        target_i_rgb = np.concatenate(
            [target_i[..., 0:1] / 9., target_i[..., 1:2] / 4., target_i[..., 2:3] / 9.],
            axis=-1)
        target_i = np.concatenate(
            [target_i[..., 0:1] / 9. * 2.0 * np.pi, target_i[..., 1:2] / 4. * np.pi, target_i[..., 2:3] / 9.* 2.0 * np.pi],
            axis=-1)

        ymis_ori = cal_misori(sam_i[..., x.shape[2] // 2, :, :3],
                              target_i[..., x.shape[2] // 2, :,
                              :3])  # misorientation angle
        yfil = (ymis_ori > 15.0) & (ymis_ori < 75.0)
        yx_img = sam_i_rgb[..., x.shape[2] // 2, :, :3]
        yy_img = target_i_rgb[..., x.shape[2] // 2, :, :3]
        ydif_img = cmap(ymis_ori * yfil / 90.0)[..., :3]
        acc.append(np.sum(yfil==0)/x.shape[-2]/x.shape[-4]*100)
        if ea_i == 0:
            show_img = np.rot90(np.vstack((yx_img, yy_img, ydif_img)), 1, (0, 1))
        else:
            show_img = np.vstack((show_img, np.rot90(np.vstack((yx_img, yy_img, ydif_img)), 1, (0, 1))))
    ave_acc = format(np.mean(np.array(acc)), '.1f')
    fig_name = folder_name + '/cos_com_' + note + '_epoch' + str(epoch)+ '_acc' + str(ave_acc)  + '.jpg'
    plt.imsave(fig_name, show_img)


def cal_acc(x_in, y_in, ca):
    # turn one hot code into angles
    x = ca.pred_angle(x_in)
    acc = []
    nx = x.shape[1]
    ny = x.shape[2]
    nz = x.shape[3]
    x = x * (x[..., 4:5] > 1e-10)

    x_true = (y_in.permute(0, 3, 4, 2, 1)).detach().cpu().numpy()
    x_true = np.concatenate(
        [x_true[..., 0:1] / 9. * 2.0 * np.pi, x_true[..., 1:2] / 4. * np.pi, x_true[..., 2:3] / 9. * 2.0 * np.pi,
         x_true[..., 3:]], axis=-1)

    # calculate the difference between NCA and CA
    for i in range(len(x)):
        for j in range(ny // 4, ny, ny // 4):
            mis_ori = cal_misori(x[i, ..., j,:, :3],
                                 x_true[i, ..., j,:, :3])  # misorientation angle
            filter = (mis_ori > 15.0) & (mis_ori < 75.0)  # diff>10.0
            # store the rsme and accuracy in the middle part of the domain
            acc.append(1.0 - np.sum(filter) / (nz * nx))
    acc = np.array(acc) * 100
    acc_ave = np.average(acc)
    return acc_ave

def cal_ori_acc(x_in, y_in, ca):
    # turn one hot code into angles
    x = ca.pred_angle(x_in)
    x = np.concatenate(
        [x[..., 0:1] * 9. / 2.0 / np.pi, x[..., 1:2] * 4. / np.pi, x[..., 2:3] * 9. / 2.0 / np.pi,
         x[..., 3:]], axis=-1)
    acc = []
    nx = x.shape[1]
    ny = x.shape[2]
    nz = x.shape[3]
    x = x * (x[..., 4:5] > 1e-10)

    x_true = (y_in.permute(0, 3, 4, 2, 1)).detach().cpu().numpy()

    # calculate the difference between NCA and CA
    acc_tmp = (x_true[..., 0]==x[...,0]).reshape(-1,1)
    acc1 = sum(acc_tmp)/acc_tmp.shape[0]
    acc_tmp = (x_true[..., 1] == x[..., 1]).reshape(-1, 1)
    acc2 = sum(acc_tmp) / acc_tmp.shape[0]
    acc_tmp = (x_true[..., 2] == x[..., 2]).reshape(-1, 1)
    acc3 = sum(acc_tmp) / acc_tmp.shape[0]
    return acc1[0], acc2[0], acc3[0]