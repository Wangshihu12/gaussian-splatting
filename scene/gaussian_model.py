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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        """
        设置高斯模型中使用的各种激活函数和变换函数
        
        这些函数定义了如何从原始参数计算出实际的物理量，
        确保参数在合理的范围内并满足物理约束。
        """
        
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            """
            从缩放和旋转参数构建3D协方差矩阵
            
            这个内部函数将缩放向量和旋转四元数转换为完整的3D协方差矩阵，
            用于定义每个高斯椭球的形状和方向。
            
            参数:
            - scaling: 缩放参数，形状为[N, 3]
            - scaling_modifier: 全局缩放修饰符
            - rotation: 旋转四元数，形状为[N, 4]
            
            返回:
            - symm: 对称协方差矩阵的压缩表示，形状为[N, 6]
            """
            # 构建缩放-旋转矩阵L，其中L = R * S
            # R是旋转矩阵，S是缩放矩阵
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            
            # 计算协方差矩阵：Σ = L * L^T
            # 这确保协方差矩阵是正定的
            actual_covariance = L @ L.transpose(1, 2)
            
            # 提取对称矩阵的上三角部分进行压缩存储
            # 由于协方差矩阵是对称的，只需存储6个独立元素
            symm = strip_symmetric(actual_covariance)
            return symm
        
        # 缩放参数的激活函数：使用指数函数确保缩放值始终为正
        # 原始参数可以是任意实数，通过exp变换后变为正数
        self.scaling_activation = torch.exp
        
        # 缩放参数的逆激活函数：对数函数，用于将正数缩放值转换回原始参数空间
        self.scaling_inverse_activation = torch.log

        # 协方差矩阵的构建函数
        self.covariance_activation = build_covariance_from_scaling_rotation

        # 不透明度的激活函数：sigmoid函数将实数映射到[0,1]区间
        # 确保不透明度值在有效范围内
        self.opacity_activation = torch.sigmoid
        
        # 不透明度的逆激活函数：logit函数，sigmoid的反函数
        self.inverse_opacity_activation = inverse_sigmoid

        # 旋转四元数的激活函数：L2归一化确保四元数的单位长度
        # 单位四元数才能正确表示旋转
        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, optimizer_type="default"):
        """
        初始化高斯模型
        
        参数:
        - sh_degree: 球谐函数的最大阶数，控制颜色表示的复杂度
        - optimizer_type: 优化器类型，默认为"default"，也可以是"sparse_adam"
        """
        
        # 球谐函数相关参数
        self.active_sh_degree = 0              # 当前激活的球谐函数阶数，训练时会逐步增加
        self.optimizer_type = optimizer_type    # 优化器类型
        self.max_sh_degree = sh_degree         # 最大球谐函数阶数
        
        # 高斯点的基本几何和外观属性
        # 初始化为空张量，后续会通过数据加载或随机初始化填充
        self._xyz = torch.empty(0)             # 3D位置坐标，形状为[N, 3]
        self._features_dc = torch.empty(0)     # 球谐函数的直流分量（0阶），形状为[N, 1, 3]
        self._features_rest = torch.empty(0)   # 球谐函数的其余系数（1阶及以上），形状为[N, (max_degree+1)^2-1, 3]
        self._scaling = torch.empty(0)         # 缩放参数（原始值，需要激活函数变换），形状为[N, 3]
        self._rotation = torch.empty(0)        # 旋转四元数（原始值，需要归一化），形状为[N, 4]
        self._opacity = torch.empty(0)         # 不透明度（原始值，需要sigmoid变换），形状为[N, 1]
        
        # 训练过程中的辅助信息
        self.max_radii2D = torch.empty(0)      # 每个高斯点在图像空间的最大半径，用于密集化
        self.xyz_gradient_accum = torch.empty(0)  # 位置梯度的累积，用于密集化决策
        self.denom = torch.empty(0)            # 梯度累积的分母，用于归一化
        
        # 优化器和训练参数
        self.optimizer = None                  # 优化器对象，后续初始化
        self.percent_dense = 0                 # 密集化百分比
        self.spatial_lr_scale = 0              # 空间学习率缩放因子
        
        # 设置激活函数
        self.setup_functions()

    def capture(self):
        """
        捕获当前模型的完整状态
        
        这个方法用于保存模型的所有参数和状态信息，
        以便后续可以完全恢复模型的状态。
        
        返回:
        - 包含所有模型状态的元组，可用于保存检查点
        """
        return (
            self.active_sh_degree,      # 当前激活的球谐函数阶数
            self._xyz,                  # 3D位置参数
            self._features_dc,          # 球谐函数直流分量
            self._features_rest,        # 球谐函数其余系数
            self._scaling,              # 缩放参数
            self._rotation,             # 旋转参数
            self._opacity,              # 不透明度参数
            self.max_radii2D,           # 最大2D半径
            self.xyz_gradient_accum,    # 位置梯度累积
            self.denom,                 # 分母累积
            self.optimizer.state_dict(), # 优化器状态字典
            self.spatial_lr_scale,      # 空间学习率缩放
        )
    
    def restore(self, model_args, training_args):
        """
        从保存的状态恢复高斯模型
        
        这个方法用于从检查点或保存的模型状态中完全恢复模型，
        包括所有参数、优化器状态和训练相关信息。
        
        参数:
        - model_args: 模型参数元组，通常来自capture()方法的输出
        - training_args: 训练参数，用于重新设置优化器
        """
        # 解包模型参数元组，按照capture()方法中定义的顺序
        (self.active_sh_degree,     # 当前激活的球谐函数阶数
        self._xyz,                  # 3D位置参数
        self._features_dc,          # 球谐函数直流分量
        self._features_rest,        # 球谐函数其余系数
        self._scaling,              # 缩放参数（原始值）
        self._rotation,             # 旋转参数（原始四元数）
        self._opacity,              # 不透明度参数（原始值）
        self.max_radii2D,           # 2D最大半径
        xyz_gradient_accum,         # 位置梯度累积
        denom,                      # 梯度累积分母
        opt_dict,                   # 优化器状态字典
        self.spatial_lr_scale) = model_args  # 空间学习率缩放
        
        # 重新设置训练环境（创建新的优化器等）
        self.training_setup(training_args)
        
        # 恢复梯度累积信息（用于密集化决策）
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        
        # 加载优化器的历史状态（包括动量等信息）
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        """
        获取经过激活函数处理的缩放参数
        
        返回实际的缩放值（正数），而不是原始的参数值。
        通过指数激活函数确保缩放值始终为正。
        
        返回:
        - 缩放参数，形状为[N, 3]，每个高斯椭球在x、y、z方向的缩放
        """
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        """
        获取经过激活函数处理的旋转参数
        
        返回归一化后的旋转四元数，确保其为单位四元数。
        单位四元数才能正确表示3D旋转。
        
        返回:
        - 归一化的旋转四元数，形状为[N, 4]
        """
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        """
        获取3D位置坐标
        
        位置参数不需要激活函数处理，直接返回原始值。
        
        返回:
        - 3D位置坐标，形状为[N, 3]
        """
        return self._xyz

    @property
    def get_features(self):
        """
        获取完整的球谐函数特征
        
        将直流分量（0阶）和其余系数（1阶及以上）拼接成完整的特征向量。
        这些特征用于表示每个高斯点的视角相关颜色。
        
        返回:
        - 完整的球谐函数特征，形状为[N, (max_degree+1)^2, 3]
        """
        features_dc = self._features_dc      # 直流分量（基础颜色）
        features_rest = self._features_rest  # 其余系数（视角相关部分）
        # 在第1维度上拼接，得到完整的球谐函数系数
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_features_dc(self):
        """
        获取球谐函数的直流分量（0阶系数）
        
        直流分量代表高斯点的基础颜色，不依赖于视角。
        
        返回:
        - 直流分量，形状为[N, 1, 3]，表示RGB基础颜色
        """
        return self._features_dc

    @property
    def get_features_rest(self):
        """
        获取球谐函数的其余系数（1阶及以上）
        
        这些系数用于表示颜色的视角相关变化，
        实现更真实的材质和光照效果。
        
        返回:
        - 其余球谐系数，形状为[N, (max_degree+1)^2-1, 3]
        """
        return self._features_rest

    @property
    def get_opacity(self):
        """
        获取经过激活函数处理的不透明度
        
        通过sigmoid激活函数将原始参数映射到[0,1]区间，
        确保不透明度值在有效范围内。
        
        返回:
        - 不透明度值，形状为[N, 1]，范围[0,1]
        """
        return self.opacity_activation(self._opacity)

    @property
    def get_exposure(self):
        """
        获取曝光参数
        
        用于训练时的曝光校正，可以补偿不同图像间的曝光差异。
        
        返回:
        - 曝光参数张量
        """
        return self._exposure

    def get_exposure_from_name(self, image_name):
        """
        根据图像名称获取对应的曝光参数
        
        不同的训练图像可能需要不同的曝光校正参数。
        这个方法根据图像名称返回相应的曝光矩阵。
        
        参数:
        - image_name: 图像文件名
        
        返回:
        - 对应的曝光变换矩阵，形状为[4, 4]或[3, 4]
        """
        if self.pretrained_exposures is None:
            # 如果没有预训练的曝光参数，使用训练中学习的参数
            # 通过映射字典找到对应的曝光参数索引
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            # 如果有预训练的曝光参数，直接使用
            return self.pretrained_exposures[image_name]

    def get_covariance(self, scaling_modifier = 1):
        """
        获取3D协方差矩阵
        
        根据缩放和旋转参数计算每个高斯椭球的3D协方差矩阵。
        协方差矩阵定义了高斯分布的形状和方向。
        
        参数:
        - scaling_modifier: 缩放修饰符，默认为1，可用于全局调整大小
        
        返回:
        - 协方差矩阵的压缩表示，形状为[N, 6]
        （利用对称性只存储上三角部分的6个独立元素）
        """
        # 调用协方差激活函数，传入缩放参数、修饰符和旋转参数
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        """
        增加球谐函数的阶数
        
        这是一个渐进训练策略：从低阶球谐函数开始训练，
        随着训练进行逐步增加阶数，提高颜色表示的复杂度。
        
        这种策略有助于训练稳定性和收敛性。
        """
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1  # 增加当前激活的球谐函数阶数
            # 注意：不会超过预设的最大阶数

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        """
        从点云数据创建高斯模型
        
        这个方法将初始的3D点云转换为高斯点云表示，设置所有必要的参数。
        这是模型初始化的核心步骤，决定了训练的起始状态。
        
        参数:
        - pcd: BasicPointCloud对象，包含点的位置和颜色信息
        - cam_infos: 相机信息列表，用于设置曝光参数
        - spatial_lr_scale: 空间学习率缩放因子
        """
        self.spatial_lr_scale = spatial_lr_scale
        
        # 将点云的3D坐标转换为CUDA张量
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        
        # 将RGB颜色转换为球谐函数的直流分量
        # RGB2SH函数将RGB颜色转换为球谐函数的0阶系数
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        
        # 初始化球谐函数特征张量
        # 形状为[N, 3, (max_degree+1)^2]，其中N是点的数量
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        
        # 设置球谐函数的直流分量（0阶系数）为转换后的颜色
        features[:, :3, 0 ] = fused_color
        # 将其余高阶系数初始化为0
        features[:, 3:, 1:] = 0.0

        print("初始化时的点数量: ", fused_point_cloud.shape[0])

        # 计算每个点到最近邻点的距离，用于设置初始缩放
        # distCUDA2是CUDA加速的距离计算函数
        # clamp_min确保距离不会太小，避免数值不稳定
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        
        # 根据最近邻距离设置初始缩放参数
        # 使用对数空间的缩放，因为后续会通过exp激活函数变换
        # 在三个方向上使用相同的缩放值
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        
        # 初始化旋转四元数为单位四元数（无旋转）
        # [1, 0, 0, 0] 表示单位四元数
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        # 初始化不透明度为0.1（在logit空间中）
        # 使用inverse_opacity_activation将[0,1]的值转换为原始参数空间
        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # 将所有参数设置为可训练的PyTorch参数
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        # 重新排列特征维度：从[N, 3, (degree+1)^2]到[N, (degree+1)^2, 3]
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        
        # 初始化2D最大半径记录（用于密集化）
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
        # 创建图像名称到索引的映射，用于曝光参数管理
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        
        # 初始化曝光参数为单位变换矩阵
        # 形状为[N_cameras, 3, 4]，每个相机一个曝光变换矩阵
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        """
        设置训练环境，包括优化器和学习率调度器
        
        参数:
        - training_args: 训练参数对象，包含各种学习率和优化器设置
        """
        # 设置密集化阈值
        self.percent_dense = training_args.percent_dense
        
        # 初始化梯度累积器（用于密集化决策）
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # 为不同类型的参数设置不同的学习率
        # 这种分组优化策略对训练稳定性很重要
        l = [
            # 位置参数：使用空间缩放的学习率
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            # 直流分量：基础颜色的学习率
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            # 其余球谐系数：使用较小的学习率（1/20）
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            # 不透明度参数
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            # 缩放参数
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            # 旋转参数
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        # 根据优化器类型创建相应的优化器
        if self.optimizer_type == "default":
            # 使用标准Adam优化器
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            # 尝试使用稀疏Adam优化器（如果可用）
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # 如果稀疏Adam不可用，回退到标准Adam
                # 需要特殊版本的光栅化器才能启用稀疏Adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        # 为曝光参数创建单独的优化器
        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        # 设置位置参数的指数学习率调度器
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init*self.spatial_lr_scale,    # 初始学习率
            lr_final=training_args.position_lr_final*self.spatial_lr_scale,  # 最终学习率
            lr_delay_mult=training_args.position_lr_delay_mult,              # 延迟乘数
            max_steps=training_args.position_lr_max_steps                    # 最大步数
        )
        
        # 设置曝光参数的指数学习率调度器
        self.exposure_scheduler_args = get_expon_lr_func(
            training_args.exposure_lr_init,                     # 初始学习率
            training_args.exposure_lr_final,                    # 最终学习率
            lr_delay_steps=training_args.exposure_lr_delay_steps,  # 延迟步数
            lr_delay_mult=training_args.exposure_lr_delay_mult,    # 延迟乘数
            max_steps=training_args.iterations                     # 最大迭代次数
        )

    def update_learning_rate(self, iteration):
        """
        根据当前迭代次数更新学习率
        
        实现动态学习率调度，通常使用指数衰减策略。
        
        参数:
        - iteration: 当前迭代次数
        
        返回:
        - 当前位置参数的学习率
        """
        # 更新曝光参数的学习率（如果没有预训练曝光）
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        # 更新主优化器中各参数组的学习率
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                # 特别处理位置参数的学习率
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr  # 返回位置参数的学习率用于日志记录

    def construct_list_of_attributes(self):
        """
        构建属性名称列表，用于PLY文件保存
        
        生成所有模型参数的标准化名称，确保保存和加载的一致性。
        
        返回:
        - 属性名称列表
        """
        # 基础属性：位置坐标和法向量占位符
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        
        # 添加直流分量特征的名称
        # 遍历所有直流分量的元素
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        
        # 添加其余球谐系数特征的名称
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        
        # 添加不透明度属性
        l.append('opacity')
        
        # 添加缩放参数的名称（通常是3个：x, y, z方向）
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        
        # 添加旋转参数的名称（四元数的4个分量）
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        
        return l

    def save_ply(self, path):
        """
        将高斯模型保存为PLY文件格式
        
        PLY是一种通用的3D数据格式，便于可视化和进一步处理。
        
        参数:
        - path: 保存文件的路径
        """
        # 创建保存目录（如果不存在）
        mkdir_p(os.path.dirname(path))

        # 将所有参数从GPU转移到CPU并转换为numpy数组
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)  # 法向量占位符（全零）
        
        # 处理直流分量：转置并展平
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        
        # 处理其余球谐系数：转置并展平
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        
        # 处理其他参数
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        # 创建PLY文件的数据类型描述
        # 每个属性都使用32位浮点数格式
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        # 创建结构化数组来存储所有数据
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        
        # 将所有属性数据拼接成一个大数组
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        
        # 将数据填入结构化数组
        elements[:] = list(map(tuple, attributes))
        
        # 创建PLY元素描述
        el = PlyElement.describe(elements, 'vertex')
        
        # 写入PLY文件
        PlyData([el]).write(path)

    def reset_opacity(self):
        """
        重置所有高斯点的不透明度
        
        这个方法将所有高斯点的不透明度限制在一个较小的值（0.01），
        通常在训练过程中定期调用，防止某些点变得过于不透明而影响训练。
        这是一种正则化技术，有助于保持模型的可优化性。
        """
        # 将当前不透明度与0.01比较，取较小值
        # 这样确保不透明度不会超过0.01
        opacities_new = self.inverse_opacity_activation(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01)
        )
        
        # 将新的不透明度值替换到优化器中
        # 这步很重要，确保优化器的内部状态与新参数同步
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        
        # 更新模型的不透明度参数
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        """
        从PLY文件加载预训练的高斯模型
        
        这个方法用于加载之前保存的模型参数，支持继续训练或推理。
        
        参数:
        - path: PLY文件路径
        - use_train_test_exp: 是否使用训练测试曝光参数
        """
        # 读取PLY文件
        plydata = PlyData.read(path)
        
        # 如果需要使用训练测试曝光参数
        if use_train_test_exp:
            # 构建曝光参数文件的路径
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            
            if os.path.exists(exposure_file):
                # 加载预训练的曝光参数
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                
                # 将曝光参数转换为CUDA张量字典
                self.pretrained_exposures = {
                    image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() 
                    for image_name in exposures
                }
                print(f"预训练曝光参数已加载。")
            else:
                print(f"在 {exposure_file} 未找到曝光参数文件")
                self.pretrained_exposures = None

        # 从PLY文件中提取3D坐标
        # 将x, y, z坐标堆叠成[N, 3]的数组
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        
        # 提取不透明度，并添加一个维度使其成为[N, 1]
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        # 初始化直流分量特征数组[N, 3, 1]
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        # 分别提取RGB三个通道的直流分量
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])  # R通道
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])  # G通道
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])  # B通道

        # 提取其余球谐函数系数
        # 找到所有以"f_rest_"开头的属性名
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        # 按照数字索引排序，确保顺序正确
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        
        # 验证球谐系数数量是否正确
        # 总系数数量应该是 3*(max_degree+1)^2 - 3（减去直流分量的3个系数）
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        
        # 提取所有额外的球谐系数
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        # 重新整形为 (P, F, SH_coeffs except DC)
        # 其中P是点数，F是特征维度（3个RGB通道），最后一维是除直流分量外的球谐系数
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        # 提取缩放参数
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # 提取旋转参数（四元数）
        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # 将所有numpy数组转换为CUDA张量并设置为可训练参数
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        # 转置特征维度以匹配内部表示格式
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        # 设置当前激活的球谐函数阶数为最大值
        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        """
        替换优化器中指定名称的张量参数
        
        这个方法用于在训练过程中动态更新参数时，同时更新优化器的内部状态。
        这对于密集化和剪枝操作非常重要，确保优化器状态与新参数保持同步。
        
        参数:
        - tensor: 新的张量参数
        - name: 参数组的名称（如"xyz", "opacity"等）
        
        返回:
        - optimizable_tensors: 包含更新后参数的字典
        """
        optimizable_tensors = {}
        
        # 遍历优化器的所有参数组
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                # 获取旧参数的优化器状态（如动量信息）
                stored_state = self.optimizer.state.get(group['params'][0], None)
                
                # 重置优化器状态为新张量的形状
                # exp_avg和exp_avg_sq是Adam优化器的动量和二阶动量
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                # 删除旧参数的状态
                del self.optimizer.state[group['params'][0]]
                
                # 设置新的参数
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                
                # 为新参数设置优化器状态
                self.optimizer.state[group['params'][0]] = stored_state

                # 记录更新后的参数
                optimizable_tensors[group["name"]] = group["params"][0]
        
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        """
        根据掩码剪枝优化器中的参数
        
        这是一个内部方法，用于在删除某些高斯点时同步更新优化器状态。
        确保优化器的内部状态（如动量）与剪枝后的参数保持一致。
        
        参数:
        - mask: 布尔掩码，True表示保留的点，False表示删除的点
        
        返回:
        - optimizable_tensors: 包含剪枝后参数的字典
        """
        optimizable_tensors = {}
        
        # 遍历所有参数组
        for group in self.optimizer.param_groups:
            # 获取当前参数的优化器状态
            stored_state = self.optimizer.state.get(group['params'][0], None)
            
            if stored_state is not None:
                # 如果存在优化器状态，则根据掩码进行剪枝
                # 只保留mask为True的元素
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                # 删除旧的状态
                del self.optimizer.state[group['params'][0]]
                
                # 创建剪枝后的新参数
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                
                # 为新参数设置剪枝后的状态
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                # 如果没有优化器状态，直接剪枝参数
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        
        return optimizable_tensors

    def prune_points(self, mask):
        """
        剪枝指定的高斯点
        
        这个方法用于删除不需要的高斯点，通常在密集化过程中调用。
        它会同时更新所有相关的参数和优化器状态。
        
        参数:
        - mask: 布尔掩码，True表示要删除的点，False表示保留的点
        """
        # 创建有效点的掩码（与输入掩码相反）
        valid_points_mask = ~mask
        
        # 剪枝优化器中的所有参数
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        # 更新所有模型参数为剪枝后的版本
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # 同时剪枝辅助统计信息
        # 这些信息用于密集化决策和训练监控
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]  # 梯度累积
        self.denom = self.denom[valid_points_mask]                           # 分母累积
        self.max_radii2D = self.max_radii2D[valid_points_mask]               # 最大2D半径
        self.tmp_radii = self.tmp_radii[valid_points_mask]    

    def cat_tensors_to_optimizer(self, tensors_dict):
        """
        将新的张量拼接到优化器的现有参数中
        
        这个方法用于在密集化过程中添加新的高斯点时，同时更新优化器的状态。
        它确保新添加的参数也能被优化器正确管理。
        
        参数:
        - tensors_dict: 包含新张量的字典，键为参数名，值为新的张量
        
        返回:
        - optimizable_tensors: 包含拼接后参数的字典
        """
        optimizable_tensors = {}
        
        # 遍历优化器的所有参数组
        for group in self.optimizer.param_groups:
            # 确保每个参数组只有一个参数张量
            assert len(group["params"]) == 1
            
            # 获取要添加的新张量
            extension_tensor = tensors_dict[group["name"]]
            
            # 获取当前参数的优化器状态
            stored_state = self.optimizer.state.get(group['params'][0], None)
            
            if stored_state is not None:
                # 如果存在优化器状态（如Adam的动量），需要扩展这些状态
                
                # 为新张量创建零初始化的一阶动量（exp_avg）
                stored_state["exp_avg"] = torch.cat((
                    stored_state["exp_avg"], 
                    torch.zeros_like(extension_tensor)
                ), dim=0)
                
                # 为新张量创建零初始化的二阶动量（exp_avg_sq）
                stored_state["exp_avg_sq"] = torch.cat((
                    stored_state["exp_avg_sq"], 
                    torch.zeros_like(extension_tensor)
                ), dim=0)

                # 删除旧参数的状态
                del self.optimizer.state[group['params'][0]]
                
                # 将原参数与新张量拼接，创建新的参数
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                
                # 为新参数设置扩展后的优化器状态
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                # 如果没有优化器状态，直接拼接参数
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        """
        密集化后处理函数
        
        这个方法在添加新的高斯点后执行清理工作，包括更新所有参数
        和重置统计信息以准备下一轮密集化。
        
        参数:
        - new_xyz: 新点的3D坐标
        - new_features_dc: 新点的直流分量特征
        - new_features_rest: 新点的其余球谐系数
        - new_opacities: 新点的不透明度
        - new_scaling: 新点的缩放参数
        - new_rotation: 新点的旋转参数
        - new_tmp_radii: 新点的临时半径
        """
        # 组织新参数为字典格式
        d = {"xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation}

        # 将新参数拼接到优化器中
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        
        # 更新模型的所有参数为拼接后的版本
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # 拼接临时半径信息
        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        
        # 重置所有点的密集化统计信息
        # 这是因为添加了新点，所有统计都需要重新开始
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")  # 梯度累积
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")              # 分母累积
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")          # 最大2D半径

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        """
        密集化并分裂大的高斯点
        
        这个方法用于处理那些梯度大且尺寸也大的高斯点。
        它将这些点分裂成多个较小的点，以提高细节表示能力。
        
        参数:
        - grads: 梯度信息
        - grad_threshold: 梯度阈值
        - scene_extent: 场景范围
        - N: 分裂数量，默认为2（每个点分裂成2个）
        """
        n_init_points = self.get_xyz.shape[0]
        
        # 创建填充后的梯度张量，处理尺寸不匹配的情况
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        
        # 选择满足梯度条件的点
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        
        # 进一步筛选：只选择那些最大缩放值大于场景范围一定比例的点
        # 这些是"大"的高斯点，适合分裂
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent
        )

        # 获取选中点的标准差（缩放参数）并重复N次
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        
        # 创建零均值的采样点
        means = torch.zeros((stds.size(0), 3), device="cuda")
        
        # 从正态分布中采样新的偏移位置
        samples = torch.normal(mean=means, std=stds)
        
        # 获取选中点的旋转矩阵
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        
        # 计算新点的3D坐标：原点 + 旋转后的采样偏移
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        
        # 缩小新点的尺寸：除以(0.8*N)，使分裂后的点更小
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        
        # 复制其他属性
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        # 执行密集化后处理，添加新点
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, 
                                new_opacity, new_scaling, new_rotation, new_tmp_radii)

        # 创建剪枝过滤器：删除原来的大点，保留新分裂的小点
        prune_filter = torch.cat((
            selected_pts_mask,  # 标记原来要分裂的点为删除
            torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)  # 新点不删除
        ))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        """
        密集化并克隆小的高斯点
        
        这个方法用于处理那些梯度大但尺寸小的高斯点。
        它直接克隆这些点，增加该区域的点密度。
        
        参数:
        - grads: 梯度信息
        - grad_threshold: 梯度阈值
        - scene_extent: 场景范围
        """
        # 选择梯度大于阈值的点
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        
        # 进一步筛选：只选择那些最大缩放值小于等于场景范围一定比例的点
        # 这些是"小"的高斯点，适合克隆
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent
        )
        
        # 直接复制选中点的所有属性
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        # 执行密集化后处理，添加克隆的点
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, 
                                new_opacities, new_scaling, new_rotation, new_tmp_radii)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        """
        执行密集化和剪枝的主要方法
        
        这是密集化算法的核心，它结合了点的添加（通过克隆和分裂）
        和删除（通过剪枝），以优化点云的分布。
        
        参数:
        - max_grad: 最大梯度阈值
        - min_opacity: 最小不透明度阈值
        - extent: 场景范围
        - max_screen_size: 最大屏幕尺寸
        - radii: 点的半径信息
        """
        # 计算平均梯度：累积梯度除以累积次数
        grads = self.xyz_gradient_accum / self.denom
        # 处理NaN值（可能由于除零产生）
        grads[grads.isnan()] = 0.0

        # 保存当前的半径信息
        self.tmp_radii = radii
        
        # 先执行克隆操作（处理小点）
        self.densify_and_clone(grads, max_grad, extent)
        
        # 再执行分裂操作（处理大点）
        self.densify_and_split(grads, max_grad, extent)

        # 创建剪枝掩码：删除不透明度过低的点
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        
        # 如果设置了最大屏幕尺寸，还要删除过大的点
        if max_screen_size:
            # 在屏幕空间中过大的点
            big_points_vs = self.max_radii2D > max_screen_size
            # 在世界空间中过大的点
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            # 合并所有剪枝条件
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        
        # 执行点的剪枝
        self.prune_points(prune_mask)
        
        # 清理临时变量
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        # 清理GPU缓存
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        """
        添加密集化统计信息
        
        这个方法在每次渲染后调用，累积用于密集化决策的统计信息。
        主要记录每个点的梯度信息，用于判断哪些区域需要更多的点。
        
        参数:
        - viewspace_point_tensor: 视图空间中的点张量（包含梯度信息）
        - update_filter: 更新过滤器，指示哪些点需要更新统计
        """
        # 累积位置梯度的L2范数
        # 只考虑前两个维度（x, y屏幕坐标）的梯度
        # 梯度大的地方表示该区域对渲染质量影响大，需要更多点
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], 
            dim=-1, 
            keepdim=True
        )
        
        # 累积更新次数，用作分母进行平均
        self.denom[update_filter] += 1
