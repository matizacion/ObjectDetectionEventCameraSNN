import argparse
import math
import os
import random
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import snntorch as snn
from snntorch import utils, surrogate

try:
    import comet_ml  # must be imported before torch (if installed)
except ImportError:
    comet_ml = None

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm

pat = #
FILE = Path(pat).resolve()
ROOT = FILE.parents[0]  # YOLOv3 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val as validate  # for end-of-epoch mAP
from models.experimental import attempt_load
# from models.yolo import Model

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None
from utils.plots import feature_visualization

from utils.autoanchor import check_anchors, check_anchor_order
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.downloads import attempt_download, is_url
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    check_amp,
    check_dataset,
    check_file,
    check_git_info,
    check_git_status,
    check_img_size,
    check_requirements,
    check_suffix,
    check_yaml,
    colorstr,
    get_latest_run,
    increment_path,
    init_seeds,
    intersect_dicts,
    labels_to_class_weights,
    labels_to_image_weights,
    methods,
    one_cycle,
    print_args,
    print_mutation,
    strip_optimizer,
    yaml_save,
)
from utils.loggers import Loggers
from utils.loggers.comet.comet_utils import check_comet_resume
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.torch_utils import (
    EarlyStopping,
    ModelEMA,
    initialize_weights,
    de_parallel,
    select_device,
    smart_DDP,
    smart_optimizer,
    smart_resume,
    torch_distributed_zero_first,
    time_sync,
    fuse_conv_and_bn,
    model_info,
)
from models.common import *


class Detect(nn.Module):
    """YOLOv3 Detect head for processing detection model outputs, including grid and anchor grid generation."""

    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        """Initializes YOLOv3 detection layer with class count, anchors, channels, and operation modes."""
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x):
        """
        Processes input through convolutional layers, reshaping output for detection.

        Expects x as list of tensors with shape(bs, C, H, W).
        """
        z = []  # inference output
        # self.stride = self.stride.to(x[0].device)
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                self.grid[i] = self.grid[i].to(x[i].device)
                self.anchor_grid[i] = self.anchor_grid[i].to(x[i].device)
                # print('#xkurdeD', x[i].device)
                self.anchor_grid[i].to(x[i].device)
                # print(self.grid[i].device, self.stride[i].device)
                # print(xy.device, wh.device, conf.device)
                xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, "1.10.0")):
        """Generates a grid and corresponding anchor grid with shape `(1, num_anchors, ny, nx, 2)` for indexing
        anchors.
        """
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        #print(y.device, x.device)
        yv, xv = torch.meshgrid(y, x, indexing="ij") if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid
    
class CNNEMS(nn.Module):
    def __init__(self):
        super(CNNEMS, self).__init__()
        self.stride = None
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.SiLU())
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.SiLU())
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.SiLU())
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.SiLU())
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.SiLU())
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.SiLU())
        
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.SiLU())
        
        self.conv8 = nn.Sequential(
            nn.Conv2d(256, 128, 1, 1),
            nn.BatchNorm2d(128),
            nn.SiLU())
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv9 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.SiLU())
    
        self.detect = Detect(2, [[10, 14, 23, 27, 37, 58], [81, 82, 135, 169, 344, 319]], [256, 512])
        
    def forward(self, x):
        x1 = self.conv4(self.conv3(self.conv2(self.conv1(x))))
        x2 = self.conv6(self.conv5(x1))
        det1 = self.conv7(x2)
        x3 = self.upsample(self.conv8(x2))
        x3 = torch.cat((x1, x3), 1)
        det2 = self.conv9(x3)
        return self.detect([det2, det1])

class EMSOneSnnLayer(nn.Module):
    def __init__(self):
        super(EMSOneSnnLayer, self).__init__()
        self.stride = None
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.SiLU())
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.SiLU())
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.SiLU())
        
        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)
        self.lif = snn.Leaky(0.8)
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.SiLU())
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.SiLU())
        
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.SiLU())
        
        self.conv8 = nn.Sequential(
            nn.Conv2d(256, 128, 1, 1),
            nn.BatchNorm2d(128),
            nn.SiLU())
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv9 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.SiLU())
    
        self.detect = Detect(2, [[10, 14, 23, 27, 37, 58], [81, 82, 135, 169, 344, 319]], [256, 512])
        
    def forward(self, x):
        mem1 = self.lif.reset_mem()
        spk, mem1 = self.lif(self.conv3(self.conv2(self.conv1(x))), mem1)

        x1 = self.conv4(spk)
        x2 = self.conv6(self.conv5(x1))
        det1 = self.conv7(x2)
        x3 = self.upsample(self.conv8(x2))
        x3 = torch.cat((x1, x3), 1)
        det2 = self.conv9(x3)
        return self.detect([det2, det1])
    
class EMSTwoSnnLayer(nn.Module):
    def __init__(self):
        super(EMSTwoSnnLayer, self).__init__()
        self.stride = None
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.SiLU())
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.SiLU())
        
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.lif3 = snn.Leaky(0.8)

        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)
        self.lif4 = snn.Leaky(0.8)
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.SiLU())
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.SiLU())
        
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.SiLU())
        
        self.conv8 = nn.Sequential(
            nn.Conv2d(256, 128, 1, 1),
            nn.BatchNorm2d(128),
            nn.SiLU())
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv9 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.SiLU())
    
        self.detect = Detect(2, [[10, 14, 23, 27, 37, 58], [81, 82, 135, 169, 344, 319]], [256, 512])
        
    def forward(self, x):
        mem4 = self.lif4.reset_mem()
        mem3 = self.lif3.reset_mem()

        spk3, mem3 = self.lif3(self.conv2(self.conv1(x)),mem3)
        cur4 = self.conv3(spk3)
        spk4, mem4 = self.lif4(cur4, mem4)

        x1 = self.conv4(spk4)
        x2 = self.conv6(self.conv5(x1))
        det1 = self.conv7(x2)
        x3 = self.upsample(self.conv8(x2))
        x3 = torch.cat((x1, x3), 1)
        det2 = self.conv9(x3)
        return self.detect([det2, det1])

class EMSSnnBackbone(nn.Module):
    def __init__(self):
        super(EMSSnnBackbone, self).__init__()
        self.stride = None
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.SiLU())
        
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.lif2 = snn.Leaky(0.8)

        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.lif3 = snn.Leaky(0.8)

        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)
        self.lif4 = snn.Leaky(0.8)
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.SiLU())
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.SiLU())
        
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.SiLU())
        
        self.conv8 = nn.Sequential(
            nn.Conv2d(256, 128, 1, 1),
            nn.BatchNorm2d(128),
            nn.SiLU())
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv9 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.SiLU())
    
        self.detect = Detect(2, [[10, 14, 23, 27, 37, 58], [81, 82, 135, 169, 344, 319]], [256, 512])
        
    def forward(self, x):
        mem4 = self.lif4.reset_mem()
        mem3 = self.lif3.reset_mem()
        mem2 = self.lif2.reset_mem()

        spk2, mem2 = self.lif2(self.conv1(x),mem2)
        cur3 = self.conv2(spk2)
        spk3, mem3 = self.lif3(cur3,mem3)
        cur4 = self.conv3(spk3)
        spk4, mem4 = self.lif4(cur4, mem4)

        x1 = self.conv4(spk4)
        x2 = self.conv6(self.conv5(x1))
        det1 = self.conv7(x2)
        x3 = self.upsample(self.conv8(x2))
        x3 = torch.cat((x1, x3), 1)
        det2 = self.conv9(x3)
        return self.detect([det2, det1])
    

class EMSSnnBackbone_1(nn.Module):
    def __init__(self, time_steps = 3, membrane_decay=0.8, learn_threshold=False, learn_beta = False, grad_func = None):
        super(EMSSnnBackbone_1, self).__init__()
        self.stride = None
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.SiLU())
        self.membrane_decay = membrane_decay
        self.time_steps = time_steps
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.lif2 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)

        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.lif3 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)

        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)
        self.lif4 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.SiLU())
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.SiLU())
        
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.SiLU())
        
        self.conv8 = nn.Sequential(
            nn.Conv2d(256, 128, 1, 1),
            nn.BatchNorm2d(128),
            nn.SiLU())
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv9 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.SiLU())
    
        self.detect = Detect(2, [[10, 14, 23, 27, 37, 58], [81, 82, 135, 169, 344, 319]], [256, 512])
        
    def forward(self, x):
        mem4 = self.lif4.reset_mem()
        mem3 = self.lif3.reset_mem()
        mem2 = self.lif2.reset_mem()

        for _ in range(self.time_steps):
            spk2, mem2 = self.lif2(self.conv1(x),mem2)
            cur3 = self.conv2(spk2)
            spk3, mem3 = self.lif3(cur3,mem3)
            cur4 = self.conv3(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)

        x1 = self.conv4(spk4)
        x2 = self.conv6(self.conv5(x1))
        det1 = self.conv7(x2)
        x3 = self.upsample(self.conv8(x2))
        x3 = torch.cat((x1, x3), 1)
        det2 = self.conv9(x3)
        return self.detect([det2, det1])
    

class CNNEMS(nn.Module):
    def __init__(self):
        super(CNNEMS, self).__init__()
        self.stride = None
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.SiLU())
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.SiLU())
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.SiLU())
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.SiLU())
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.SiLU())
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.SiLU())
        
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.SiLU())
        
        self.conv8 = nn.Sequential(
            nn.Conv2d(256, 128, 1, 1),
            nn.BatchNorm2d(128),
            nn.SiLU())
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv9 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.SiLU())
    
        self.detect = Detect(2, [[10, 14, 23, 27, 37, 58], [81, 82, 135, 169, 344, 319]], [256, 512])
        
    def forward(self, x):
        x1 = self.conv4(self.conv3(self.conv2(self.conv1(x))))
        x2 = self.conv6(self.conv5(x1))
        det1 = self.conv7(x2)
        x3 = self.upsample(self.conv8(x2))
        x3 = torch.cat((x1, x3), 1)
        det2 = self.conv9(x3)
        return self.detect([det2, det1])
    
class EMSSixSnnLayer(nn.Module):
    def __init__(self, time_steps=3, membrane_decay=0.8, learn_threshold=False, learn_beta = False, grad_func = None):
        super(EMSSixSnnLayer, self).__init__()
        self.membrane_decay = membrane_decay
        self.time_steps = time_steps
        self.stride = None
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.SiLU())
        
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.lif2 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)

        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.lif3 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)

        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)
        self.lif4 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)
        
        self.conv5 = nn.Conv2d(256, 512, 3, 2, 1)
        self.lif5 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)

        self.conv6 = nn.Conv2d(512, 256, 3, 1, 1)
        self.lif6 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)
        
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.SiLU())
        
        self.conv8 = nn.Conv2d(256, 128, 1, 1)
        self.lif8 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv9 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.SiLU())
    
        self.detect = Detect(2, [[10, 14, 23, 27, 37, 58], [81, 82, 135, 169, 344, 319]], [256, 512])
        
    def forward(self, x):
        mem8 = self.lif8.reset_mem()
        mem6 = self.lif6.reset_mem()
        mem5 = self.lif5.reset_mem()
        mem4 = self.lif4.reset_mem()
        mem3 = self.lif3.reset_mem()
        mem2 = self.lif2.reset_mem()

        for _ in range(self.time_steps):
            spk2, mem2 = self.lif2(self.conv1(x),mem2)
            out2 = self.conv2(spk2)
            spk3, mem3 = self.lif3(out2,mem3)
            out3 = self.conv3(spk3)
            spk4, mem4 = self.lif4(out3,mem4)
            out4 = self.conv4(spk4)
            spk5, mem5 = self.lif5(out4,mem5)
            out5 = self.conv5(spk5)
            spk6, mem6 = self.lif6(out5,mem6)
            out6 = self.conv6(spk6)
            
            det1 = self.conv7(out6)
            spk8, mem8 = self.lif8(out6,mem8)
            out8 = self.conv8(spk8)

            outUpsample = self.upsample(out8)
            outUpsample = torch.cat((out4, outUpsample), 1)
            det2 = self.conv9(outUpsample)
        return self.detect([det2, det1])     
    
class EMSSnnPiramide(nn.Module):
    def __init__(self, time_steps=3, membrane_decay=0.8, learn_threshold=False, learn_beta = False, grad_func = None):
        super(EMSSnnPiramide, self).__init__()

        self.stride = None
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.SiLU())
        self.time_steps = time_steps
        self.membrane_decay = membrane_decay

        if grad_func is not None:
            grad_func = grad_func

        
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.lif2 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)

        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.lif3 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)

        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)
        self.lif4 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)
        
        self.conv5 = nn.Conv2d(256, 512, 3, 2, 1)
        self.lif5 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)

        
        self.conv6 = nn.Conv2d(512, 256, 3, 1, 1)
        self.lif6 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)
        
        self.conv7 = nn.Conv2d(256, 512, 3, 1, 1)
        self.lif7 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)

        self.conv8 = nn.Conv2d(256, 128, 1, 1)
        self.lif8 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv9 = nn.Conv2d(384, 256, 3, 1, 1)
        self.lif9 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)
    
        self.detect = Detect(2, [[10, 14, 23, 27, 37, 58], [81, 82, 135, 169, 344, 319]], [256, 512])
        
    def forward(self, x):
        mem9 = self.lif9.reset_mem()
        mem8 = self.lif8.reset_mem()
        mem7 = self.lif7.reset_mem()
        mem6 = self.lif6.reset_mem()
        mem5 = self.lif5.reset_mem()
        mem4 = self.lif4.reset_mem()
        mem3 = self.lif3.reset_mem()
        mem2 = self.lif2.reset_mem()
        for _ in range(self.time_steps):

            spk2, mem2 = self.lif2(self.conv1(x),mem2)
            out2 = self.conv2(spk2)
            spk3, mem3 = self.lif3(out2,mem3)
            out3 = self.conv3(spk3)
            spk4, mem4 = self.lif4(out3,mem4)
            out4 = self.conv4(spk4)
            spk5, mem5 = self.lif5(out4,mem5)
            out5 = self.conv5(spk5)
            spk6, mem6 = self.lif6(out5,mem6)
            out6 = self.conv6(spk6)
            spk7, mem7 = self.lif7(out6,mem7)
            det1 = self.conv7(spk7)


            spk8, mem8 = self.lif8(out6,mem8)
            out8 = self.conv8(spk8)

            outUpsample = self.upsample(out8)
            outUpsample = torch.cat((out4, outUpsample), 1)

            spk9, mem9 = self.lif9(outUpsample,mem9)

            det2 = self.conv9(spk9)

        return self.detect([det2, det1])

class DetectSNN_old_version(nn.Module):
    """YOLOv3 Detect head for processing detection model outputs, including grid and anchor grid generation."""

    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), membrane_decay = 0.8, learn_threshold = False, learn_beta = False, grad_func = None, time_steps = 3, inplace=True):  # detection layer
        """Initializes YOLOv3 detection layer with class count, anchors, channels, and operation modes."""
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)
        self.lif256 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)
        self.lif512 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)
        self.time_steps = time_steps
        self.lif5 = snn.Leaky(0.8)
        self.lif6 = snn.Leaky(0.8)


    def forward(self, x):
        """
        Processes input through convolutional layers, reshaping output for detection.

        Expects x as list of tensors with shape(bs, C, H, W).
        """
        z = []  # inference output
        mem5 = self.lif512.reset_mem()
        mem6 = self.lif256.reset_mem()
        for i in range(self.nl):
            for _ in range(self.time_steps):
                if i == 0:
                    _, mem5 = self.lif512(x[i], mem5)
                    o = self.m[i](mem5)  # conv
                elif i == 1:
                    _, mem6 = self.lif256(x[i], mem6)
                    o = self.m[i](mem6)  # conv  # conv
            x[i] = o
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                self.grid[i] = self.grid[i].to(x[i].device)
                self.anchor_grid[i] = self.anchor_grid[i].to(x[i].device)
                self.anchor_grid[i].to(x[i].device)
                xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, "1.10.0")):
        """Generates a grid and corresponding anchor grid with shape `(1, num_anchors, ny, nx, 2)` for indexing
        anchors.
        """
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        #print(y.device, x.device)
        yv, xv = torch.meshgrid(y, x, indexing="ij") if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid

class EMSSnn_1(nn.Module):
    def __init__(self, time_steps=3, membrane_decay=0.8, learn_threshold=False, learn_beta = False, grad_func = None):
        super(EMSSnn_1, self).__init__()
        self.membrane_decay = membrane_decay
        self.stride = None
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.SiLU())
        self.time_steps = time_steps

        if grad_func is not None:
            grad_func = grad_func

        
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.lif2 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)

        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.lif3 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)

        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)
        self.lif4 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)
        
        self.conv5 = nn.Conv2d(256, 512, 3, 2, 1)
        self.lif5 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)

        
        self.conv6 = nn.Conv2d(512, 256, 3, 1, 1)
        self.lif6 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)
        
        self.conv7 = nn.Conv2d(256, 512, 3, 1, 1)
        self.lif7 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)

        self.conv8 = nn.Conv2d(256, 128, 1, 1)
        self.lif8 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv9 = nn.Conv2d(384, 256, 3, 1, 1)
        self.lif9 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)
    
        self.detect = DetectSNN_old_version(2, [[10, 14, 23, 27, 37, 58], [81, 82, 135, 169, 344, 319]], [256, 512], membrane_decay, learn_threshold, learn_beta, grad_func, time_steps)
        
    def forward(self, x):
        mem9 = self.lif9.reset_mem()
        mem8 = self.lif8.reset_mem()
        mem7 = self.lif7.reset_mem()
        mem6 = self.lif6.reset_mem()
        mem5 = self.lif5.reset_mem()
        mem4 = self.lif4.reset_mem()
        mem3 = self.lif3.reset_mem()
        mem2 = self.lif2.reset_mem()
        for _ in range(self.time_steps):
            spk2, mem2 = self.lif2(self.conv1(x),mem2)
            out2 = self.conv2(spk2)
            spk3, mem3 = self.lif3(out2,mem3)
            out3 = self.conv3(spk3)
            spk4, mem4 = self.lif4(out3,mem4)
            out4 = self.conv4(spk4)
            spk5, mem5 = self.lif5(out4,mem5)
            out5 = self.conv5(spk5)
            spk6, mem6 = self.lif6(out5,mem6)
            out6 = self.conv6(spk6)
            spk7, mem7 = self.lif7(out6,mem7)
            det1 = self.conv7(spk7)


            spk8, mem8 = self.lif8(out6,mem8)
            out8 = self.conv8(spk8)

            outUpsample = self.upsample(out8)
            outUpsample = torch.cat((out4, outUpsample), 1)

            spk9, mem9 = self.lif9(outUpsample,mem9)

            det2 = self.conv9(spk9)

        return self.detect([det2, det1])

class DetectSnn(nn.Module):
    """YOLOv3 Detect head for processing detection model outputs, including grid and anchor grid generation."""

    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), membrane_decay = 0.8, learn_threshold = False, learn_beta = False, grad_func = None, time_steps = 3, inplace=True):  # detection layer
        """Initializes YOLOv3 detection layer with class count, anchors, channels, and operation modes."""
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)
        self.lif256 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)
        self.lif512 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)
        self.time_steps = time_steps


    def forward(self, x):
        """
        Processes input through convolutional layers, reshaping output for detection.

        Expects x as list of tensors with shape(bs, C, H, W).
        """
        z = []  # inference output
        mem5 = self.lif512.reset_mem()
        mem6 = self.lif256.reset_mem()
        for i in range(self.nl):
            for _ in range(self.time_steps):
                if i == 0:
                    _, mem5 = self.lif512(x[i], mem5)
                    o = self.m[i](mem5)  # conv
                elif i == 1:
                    _, mem6 = self.lif256(x[i], mem6)
                    o = self.m[i](mem6)  # conv  # conv
            x[i] = o
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                self.grid[i] = self.grid[i].to(x[i].device)
                self.anchor_grid[i] = self.anchor_grid[i].to(x[i].device)
                self.anchor_grid[i].to(x[i].device)
                xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, "1.10.0")):
        """Generates a grid and corresponding anchor grid with shape `(1, num_anchors, ny, nx, 2)` for indexing
        anchors.
        """
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        #print(y.device, x.device)
        yv, xv = torch.meshgrid(y, x, indexing="ij") if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid 

class EMSSnn(nn.Module):
    def __init__(self, time_steps=3, membrane_decay=0.8, learn_threshold=False, learn_beta = False, grad_func = None):
        super(EMSSnn, self).__init__()
        self.membrane_decay = membrane_decay
        self.stride = None
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.SiLU())
        self.time_steps = time_steps

        if grad_func is not None:
            grad_func = grad_func

        
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.lif2 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)

        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.lif3 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)

        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)
        self.lif4 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)
        
        self.conv5 = nn.Conv2d(256, 512, 3, 2, 1)
        self.lif5 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)

        
        self.conv6 = nn.Conv2d(512, 256, 3, 1, 1)
        self.lif6 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)
        
        self.conv7 = nn.Conv2d(256, 512, 3, 1, 1)
        self.lif7 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)

        self.conv8 = nn.Conv2d(256, 128, 1, 1)
        self.lif8 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv9 = nn.Conv2d(384, 256, 3, 1, 1)
        self.lif9 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)
    
        self.detect = DetectSnn(2, [[10, 14, 23, 27, 37, 58], [81, 82, 135, 169, 344, 319]], [256, 512], membrane_decay, learn_threshold, learn_beta, grad_func, time_steps)
        
    def forward(self, x):
        mem9 = self.lif9.reset_mem()
        mem8 = self.lif8.reset_mem()
        mem7 = self.lif7.reset_mem()
        mem6 = self.lif6.reset_mem()
        mem5 = self.lif5.reset_mem()
        mem4 = self.lif4.reset_mem()
        mem3 = self.lif3.reset_mem()
        mem2 = self.lif2.reset_mem()
        for _ in range(self.time_steps):
            spk2, mem2 = self.lif2(self.conv1(x),mem2)
            out2 = self.conv2(spk2)
            spk3, mem3 = self.lif3(out2,mem3)
            out3 = self.conv3(spk3)
            spk4, mem4 = self.lif4(out3,mem4)
            out4 = self.conv4(spk4)
            spk5, mem5 = self.lif5(out4,mem5)
            out5 = self.conv5(spk5)
            spk6, mem6 = self.lif6(out5,mem6)
            out6 = self.conv6(spk6)
            spk7, mem7 = self.lif7(out6,mem7)
            det1 = self.conv7(spk7)


            spk8, mem8 = self.lif8(out6,mem8)
            out8 = self.conv8(spk8)

            outUpsample = self.upsample(out8)
            outUpsample = torch.cat((out4, outUpsample), 1)

            spk9, mem9 = self.lif9(outUpsample,mem9)

            det2 = self.conv9(spk9)

        return self.detect([det2, det1])
    

class DetectSnnMerged(nn.Module):
    """YOLOv3 Detect head for processing detection model outputs, including grid and anchor grid generation."""

    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), membrane_decay = 0.8, learn_threshold = False, learn_beta = False, grad_func = None, time_steps = 3, inplace=True):  # detection layer
        """Initializes YOLOv3 detection layer with class count, anchors, channels, and operation modes."""
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)
        # self.lif256 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)
        # self.lif512 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)
        # self.lif = [self.lif512, self.lif256]
        self.time_steps = time_steps


    def forward(self, x):
        """
        Processes input through convolutional layers, reshaping output for detection.

        Expects x as list of tensors with shape(bs, C, H, W).
        """
        z = []  # inference output
        # mem5 = self.lif512.reset_mem()
        # mem6 = self.lif256.reset_mem()
        for i in range(self.nl):
            # for _ in range(self.time_steps):
            #     if i == 0:
            #         _, mem5 = self.lif512(x[i], mem5)
            #         o = self.m[i](mem5)  # conv
            #     elif i == 1:
            #         _, mem6 = self.lif256(x[i], mem6)
            #         o = self.m[i](mem6)  # conv  # conv
            # x[i] = o
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                self.grid[i] = self.grid[i].to(x[i].device)
                self.anchor_grid[i] = self.anchor_grid[i].to(x[i].device)
                self.anchor_grid[i].to(x[i].device)
                xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, "1.10.0")):
        """Generates a grid and corresponding anchor grid with shape `(1, num_anchors, ny, nx, 2)` for indexing
        anchors.
        """
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        #print(y.device, x.device)
        yv, xv = torch.meshgrid(y, x, indexing="ij") if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid
    

class EMSSnnMerged(nn.Module):
    def __init__(self, time_steps=3, membrane_decay=0.8, learn_threshold=False, learn_beta = False, grad_func = None):
        super(EMSSnnMerged, self).__init__()
        self.membrane_decay = membrane_decay
        self.stride = None
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.SiLU())
        self.time_steps = time_steps

        self.conv256 = nn.Conv2d(256, 21, 1) #21 = 3*7
        self.conv512 = nn.Conv2d(512, 21, 1) #21 = 3*7

        if grad_func is not None:
            grad_func = grad_func
        
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.lif2 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)

        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.lif3 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)

        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)
        self.lif4 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)
        
        self.conv5 = nn.Conv2d(256, 512, 3, 2, 1)
        self.lif5 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)

        
        self.conv6 = nn.Conv2d(512, 256, 3, 1, 1)
        self.lif6 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)
        
        self.conv7 = nn.Conv2d(256, 512, 3, 1, 1)
        self.lif7 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)

        self.conv8 = nn.Conv2d(256, 128, 1, 1)
        self.lif8 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv9 = nn.Conv2d(384, 256, 3, 1, 1)
        self.lif9 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)
    
        self.lif256 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)
        self.conv256 = nn.Conv2d(256, 21, 1) #21 = 3*7
        
        self.lif512 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)
        self.conv512 = nn.Conv2d(512, 21, 1) #21 = 3*7

        self.detect = DetectSnnMerged(2, [[10, 14, 23, 27, 37, 58], [81, 82, 135, 169, 344, 319]], [256, 512], membrane_decay, learn_threshold, learn_beta, grad_func, time_steps)
        
    def forward(self, x):
        mem512 = self.lif512.reset_mem()
        mem256 = self.lif256.reset_mem()


        mem9 = self.lif9.reset_mem()
        mem8 = self.lif8.reset_mem()
        mem7 = self.lif7.reset_mem()
        mem6 = self.lif6.reset_mem()
        mem5 = self.lif5.reset_mem()
        mem4 = self.lif4.reset_mem()
        mem3 = self.lif3.reset_mem()
        mem2 = self.lif2.reset_mem()
        for _ in range(self.time_steps):
            spk2, mem2 = self.lif2(self.conv1(x),mem2)
            out2 = self.conv2(spk2)
            spk3, mem3 = self.lif3(out2,mem3)
            out3 = self.conv3(spk3)
            spk4, mem4 = self.lif4(out3,mem4)
            out4 = self.conv4(spk4)
            spk5, mem5 = self.lif5(out4,mem5)
            out5 = self.conv5(spk5)
            spk6, mem6 = self.lif6(out5,mem6)
            out6 = self.conv6(spk6)
            spk7, mem7 = self.lif7(out6,mem7)
            det1 = self.conv7(spk7)


            spk8, mem8 = self.lif8(out6,mem8)
            out8 = self.conv8(spk8)

            outUpsample = self.upsample(out8)
            outUpsample = torch.cat((out4, outUpsample), 1)

            spk9, mem9 = self.lif9(outUpsample,mem9)

            det2 = self.conv9(spk9)

            _, mem256 = self.lif256(det2, mem256)
            _, mem512 = self.lif512(det1, mem512)

            det2 = self.conv256(mem256)
            det1 = self.conv512(mem512)

        return self.detect([det2, det1])
    

    

class EMSSnnMergedSpike(nn.Module):
    def __init__(self, time_steps=3, membrane_decay=0.8, learn_threshold=False, learn_beta = False, grad_func = None):
        super(EMSSnnMergedSpike, self).__init__()
        self.membrane_decay = membrane_decay
        self.stride = None
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.SiLU())
        self.time_steps = time_steps

        self.conv256 = nn.Conv2d(256, 21, 1) #21 = 3*7
        self.conv512 = nn.Conv2d(512, 21, 1) #21 = 3*7

        if grad_func is not None:
            grad_func = grad_func
        
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.lif2 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)

        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.lif3 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)

        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)
        self.lif4 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)
        
        self.conv5 = nn.Conv2d(256, 512, 3, 2, 1)
        self.lif5 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)

        
        self.conv6 = nn.Conv2d(512, 256, 3, 1, 1)
        self.lif6 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)
        
        self.conv7 = nn.Conv2d(256, 512, 3, 1, 1)
        self.lif7 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)

        self.conv8 = nn.Conv2d(256, 128, 1, 1)
        self.lif8 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv9 = nn.Conv2d(384, 256, 3, 1, 1)
        self.lif9 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)
    
        self.lif256 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)
        self.conv256 = nn.Conv2d(256, 21, 1) #21 = 3*7
        
        self.lif512 = snn.Leaky(membrane_decay, learn_threshold=learn_threshold, learn_beta=learn_beta, spike_grad = grad_func)
        self.conv512 = nn.Conv2d(512, 21, 1) #21 = 3*7

        self.detect = DetectSnnMerged(2, [[10, 14, 23, 27, 37, 58], [81, 82, 135, 169, 344, 319]], [256, 512], membrane_decay, learn_threshold, learn_beta, grad_func, time_steps)
        
    def forward(self, x):
        mem512 = self.lif512.reset_mem()
        mem256 = self.lif256.reset_mem()


        mem9 = self.lif9.reset_mem()
        mem8 = self.lif8.reset_mem()
        mem7 = self.lif7.reset_mem()
        mem6 = self.lif6.reset_mem()
        mem5 = self.lif5.reset_mem()
        mem4 = self.lif4.reset_mem()
        mem3 = self.lif3.reset_mem()
        mem2 = self.lif2.reset_mem()
        for _ in range(self.time_steps):
            spk2, mem2 = self.lif2(self.conv1(x),mem2)
            out2 = self.conv2(spk2)
            spk3, mem3 = self.lif3(out2,mem3)
            out3 = self.conv3(spk3)
            spk4, mem4 = self.lif4(out3,mem4)
            out4 = self.conv4(spk4)
            spk5, mem5 = self.lif5(out4,mem5)
            out5 = self.conv5(spk5)
            spk6, mem6 = self.lif6(out5,mem6)
            out6 = self.conv6(spk6)
            spk7, mem7 = self.lif7(out6,mem7)
            det1 = self.conv7(spk7)


            spk8, mem8 = self.lif8(out6,mem8)
            out8 = self.conv8(spk8)

            outUpsample = self.upsample(out8)
            outUpsample = torch.cat((out4, outUpsample), 1)

            spk9, mem9 = self.lif9(outUpsample,mem9)

            det2 = self.conv9(spk9)

            spk256, mem256 = self.lif256(det2, mem256)
            spk512, mem512 = self.lif512(det1, mem512)

            det2 = self.conv256(spk256)
            det1 = self.conv512(spk512)

        return self.detect([det2, det1])