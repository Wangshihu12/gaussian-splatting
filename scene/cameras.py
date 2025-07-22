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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch
import cv2

class Camera(nn.Module):
    """
    相机类：表示3D场景中的单个相机视角
    
    该类封装了相机的所有参数和相关数据，包括：
    - 相机的内参和外参
    - 相机拍摄的图像数据
    - 深度信息（如果可用）
    - 相机在世界坐标系中的变换矩阵
    """
    
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, image, invdepthmap,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 train_test_exp = False, is_test_dataset = False, is_test_view = False
                 ):
        """
        初始化相机对象
        
        参数:
        - resolution: 图像分辨率，格式为(width, height)
        - colmap_id: COLMAP中的相机ID，用于与COLMAP数据关联
        - R: 3x3旋转矩阵，描述相机的朝向
        - T: 3x1平移向量，描述相机的位置
        - FoVx: 水平方向的视场角（弧度）
        - FoVy: 垂直方向的视场角（弧度）
        - depth_params: 深度参数字典，包含深度缩放和偏移信息
        - image: PIL图像对象，相机拍摄的原始图像
        - invdepthmap: 逆深度图，numpy数组格式
        - image_name: 图像文件名，用于标识
        - uid: 唯一标识符
        - trans: 额外的平移变换，默认为零向量
        - scale: 场景缩放因子，默认为1.0
        - data_device: 数据存储设备，默认为"cuda"
        - train_test_exp: 是否启用训练测试曝光模式
        - is_test_dataset: 是否为测试数据集
        - is_test_view: 是否为测试视角
        """
        super(Camera, self).__init__()

        # ===== 基本相机参数设置 =====
        self.uid = uid                    # 相机唯一标识符
        self.colmap_id = colmap_id        # COLMAP系统中的相机ID
        self.R = R                        # 旋转矩阵：世界坐标系到相机坐标系
        self.T = T                        # 平移向量：相机在世界坐标系中的位置
        self.FoVx = FoVx                  # 水平视场角（Field of View X）
        self.FoVy = FoVy                  # 垂直视场角（Field of View Y）
        self.image_name = image_name      # 图像文件名

        # ===== 设备配置 =====
        try:
            # 尝试创建指定的设备对象（通常是"cuda"或"cpu"）
            self.data_device = torch.device(data_device)
        except Exception as e:
            # 如果指定设备不可用，回退到默认的CUDA设备
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        # ===== 图像数据处理 =====
        # 将PIL图像转换为PyTorch张量并调整到指定分辨率
        resized_image_rgb = PILtoTorch(image, resolution)
        gt_image = resized_image_rgb[:3, ...]  # 取RGB通道（前3个通道）
        
        # ===== Alpha通道/遮罩处理 =====
        self.alpha_mask = None
        if resized_image_rgb.shape[0] == 4:
            # 如果图像有4个通道（RGBA），第4个通道作为alpha遮罩
            self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
        else: 
            # 如果图像只有3个通道（RGB），创建全1的遮罩（表示所有像素都可见）
            self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))

        # ===== 训练测试曝光模式的特殊处理 =====
        if train_test_exp and is_test_view:
            # 这是一种特殊的训练策略，用于处理曝光变化
            if is_test_dataset:
                # 如果是测试数据集，遮蔽图像的左半部分
                self.alpha_mask[..., :self.alpha_mask.shape[-1] // 2] = 0
            else:
                # 如果是训练数据集，遮蔽图像的右半部分
                self.alpha_mask[..., self.alpha_mask.shape[-1] // 2:] = 0

        # ===== 图像数据最终处理 =====
        # 将像素值限制在[0,1]范围内，并移动到指定设备
        self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]   # 图像宽度
        self.image_height = self.original_image.shape[1]  # 图像高度

        # ===== 深度信息处理 =====
        self.invdepthmap = None      # 逆深度图
        self.depth_reliable = False  # 深度信息是否可靠
        
        if invdepthmap is not None:
            # 如果提供了深度图数据
            # 创建深度掩码，初始为全1（所有像素的深度都可信）
            self.depth_mask = torch.ones_like(self.alpha_mask)
            
            # 将深度图调整到目标分辨率
            self.invdepthmap = cv2.resize(invdepthmap, resolution)
            
            # 将负深度值设为0（深度值不能为负）
            self.invdepthmap[self.invdepthmap < 0] = 0
            
            # 标记深度信息为可靠
            self.depth_reliable = True

            # ===== 深度参数校验和调整 =====
            if depth_params is not None:
                # 检查深度缩放是否在合理范围内
                # 如果缩放因子过小或过大，认为深度不可靠
                if depth_params["scale"] < 0.2 * depth_params["med_scale"] or \
                   depth_params["scale"] > 5 * depth_params["med_scale"]:
                    self.depth_reliable = False  # 标记深度不可靠
                    self.depth_mask *= 0         # 将深度掩码置零
                
                # 应用深度缩放和偏移
                if depth_params["scale"] > 0:
                    # 深度变换：新深度 = 原深度 × 缩放 + 偏移
                    self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]

            # ===== 深度图格式转换 =====
            if self.invdepthmap.ndim != 2:
                # 如果深度图不是2D的，取第一个通道
                self.invdepthmap = self.invdepthmap[..., 0]
            
            # 转换为PyTorch张量并添加batch维度，移动到指定设备
            self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)

        # ===== 相机投影参数设置 =====
        self.zfar = 100.0   # 远裁剪面距离
        self.znear = 0.01   # 近裁剪面距离

        # ===== 场景变换参数 =====
        self.trans = trans  # 额外的平移变换
        self.scale = scale  # 场景缩放因子

        # ===== 变换矩阵计算 =====
        # 计算世界坐标系到视图坐标系的变换矩阵
        # getWorld2View2函数结合旋转R、平移T、额外平移trans和缩放scale
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        
        # 计算投影矩阵：将3D点投影到2D屏幕坐标
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, 
                                                   fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        
        # 计算完整的投影变换矩阵：世界坐标 -> 视图坐标 -> 屏幕坐标
        # 这个矩阵将3D世界坐标直接变换为2D屏幕坐标
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        
        # 计算相机在世界坐标系中的中心位置
        # 通过世界视图变换矩阵的逆矩阵的第4行前3列得到
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

