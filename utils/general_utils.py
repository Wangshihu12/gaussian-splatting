#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import sys
from datetime import datetime
import numpy as np
import random

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    """
    安全状态初始化函数：设置确定性的随机数种子并自定义输出格式
    
    参数:
    - silent: 布尔值，是否启用静默模式（不输出信息）
    
    功能:
    1. 自定义标准输出格式，为每行输出添加时间戳
    2. 设置所有随机数生成器的种子，确保实验的可重复性
    3. 设置CUDA设备
    """
    
    # ===== 保存原始标准输出 =====
    old_f = sys.stdout  # 保存原始的sys.stdout对象，用于实际的输出操作
    
    # ===== 自定义输出类 =====
    class F:
        """
        自定义输出类，用于重写标准输出的行为
        主要功能：为输出信息添加时间戳，并支持静默模式
        """
        def __init__(self, silent):
            """
            初始化自定义输出类
            
            参数:
            - silent: 是否启用静默模式
            """
            self.silent = silent  # 存储静默模式标志
            
        def write(self, x):
            """
            重写write方法，自定义输出格式
            
            参数:
            - x: 要输出的字符串
            """
            if not self.silent:  # 只有在非静默模式下才输出
                if x.endswith("\n"):  # 如果字符串以换行符结尾
                    # 在换行符前插入时间戳
                    # 时间格式：[日/月 时:分:秒]，例如：[25/12 14:30:45]
                    old_f.write(x.replace("\n", " [{}]\n".format(
                        str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    # 如果不以换行符结尾，直接输出原始内容
                    old_f.write(x)
                    
        def flush(self):
            """
            重写flush方法，确保输出缓冲区被刷新
            这对于实时输出很重要
            """
            old_f.flush()  # 调用原始输出对象的flush方法

    # ===== 替换标准输出 =====
    # 将系统的标准输出替换为我们自定义的输出类实例
    # 从此刻开始，所有的print语句都会使用我们的自定义格式
    sys.stdout = F(silent)

    # ===== 设置随机数种子以确保可重复性 =====
    # 这是机器学习实验中的重要步骤，确保每次运行得到相同的结果
    
    random.seed(0)      # 设置Python内置random模块的种子
    np.random.seed(0)   # 设置NumPy随机数生成器的种子  
    torch.manual_seed(0)  # 设置PyTorch CPU随机数生成器的种子
    
    # 注意：这里没有设置torch.cuda.manual_seed(0)，
    # 如果需要CUDA操作也完全可重复，应该添加这行代码
    
    # ===== 设置CUDA设备 =====
    # 显式设置使用第一个CUDA设备（GPU 0）
    # 这确保了在多GPU环境中有一致的设备选择
    torch.cuda.set_device(torch.device("cuda:0"))
