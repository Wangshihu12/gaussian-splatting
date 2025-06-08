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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False):
    """
    渲染场景的核心函数
    
    参数:
    - viewpoint_camera: 视点相机对象，包含相机参数和位置信息
    - pc: GaussianModel对象，包含3D高斯点云数据
    - pipe: 渲染管线参数对象
    - bg_color: 背景颜色张量，必须在GPU上
    - scaling_modifier: 缩放修饰符，默认1.0
    - separate_sh: 是否分离球谐函数处理
    - override_color: 覆盖颜色，如果提供则使用预计算颜色
    - use_trained_exp: 是否使用训练的曝光参数
    """
 
    # 创建零张量用于存储屏幕空间点坐标
    # 这个张量的梯度将被用于优化高斯点的2D屏幕位置
    # requires_grad=True 表示需要计算梯度用于反向传播
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0

    # 保留梯度，确保在反向传播时梯度不会被释放
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # 设置光栅化配置参数
    # 计算相机视场角的正切值，用于透视投影计算
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5) # 水平视场角的一半的正切值
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5) # 垂直视场角的一半的正切值

    # 创建高斯光栅化设置对象
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),  # 图像高度
        image_width=int(viewpoint_camera.image_width),    # 图像宽度
        tanfovx=tanfovx,                                  # 水平视场角正切值
        tanfovy=tanfovy,                                  # 垂直视场角正切值
        bg=bg_color,                                      # 背景颜色
        scale_modifier=scaling_modifier,                  # 缩放修饰符
        viewmatrix=viewpoint_camera.world_view_transform, # 世界到视图变换矩阵
        projmatrix=viewpoint_camera.full_proj_transform,  # 投影变换矩阵
        sh_degree=pc.active_sh_degree,                    # 当前激活的球谐函数阶数
        campos=viewpoint_camera.camera_center,            # 相机中心位置
        prefiltered=False,                                # 不使用预滤波
        debug=pipe.debug,                                 # 调试模式
        antialiasing=pipe.antialiasing                    # 抗锯齿设置
    )

    # 创建高斯光栅化器实例
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # 获取高斯点云的基本属性
    means3D = pc.get_xyz           # 3D空间中高斯点的中心位置
    means2D = screenspace_points   # 2D屏幕空间中的点位置
    opacity = pc.get_opacity       # 每个高斯点的不透明度

    # 初始化协方差相关参数
    # 如果提供了预计算的3D协方差，则使用它；否则由光栅化器从缩放/旋转计算
    scales = None          # 高斯椭球的缩放参数
    rotations = None       # 高斯椭球的旋转参数
    cov3D_precomp = None   # 预计算的3D协方差矩阵

    # 根据管线设置决定是否在Python中计算3D协方差
    if pipe.compute_cov3D_python:
        # 在Python中预计算3D协方差矩阵
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        # 让光栅化器从缩放和旋转参数计算协方差
        scales = pc.get_scaling      # 获取缩放参数
        rotations = pc.get_rotation  # 获取旋转参数

    # 处理颜色信息
    # 如果提供了预计算颜色则使用它们；否则根据需要从球谐函数计算颜色
    shs = None              # 球谐函数系数
    colors_precomp = None   # 预计算的颜色

    if override_color is None:
        # 没有提供覆盖颜色，需要从球谐函数计算颜色
        if pipe.convert_SHs_python:
            # 在Python中将球谐函数转换为RGB颜色
            # 重新整理球谐函数特征的维度：[N, (degree+1)^2, 3] -> [N, 3, (degree+1)^2]
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)

            # 计算从高斯点到相机的方向向量
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            # 归一化方向向量
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)

            # 使用球谐函数计算RGB颜色
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            # 将颜色值限制在[0, 1]范围内
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            # 让光栅化器处理球谐函数到RGB的转换
            if separate_sh:
                # 分离处理：获取直流分量和其余球谐系数
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                # 统一处理：获取所有球谐函数特征
                shs = pc.get_features
    else:
        # 使用提供的覆盖颜色
        colors_precomp = override_color

    # 将可见的高斯点光栅化到图像上，获取它们在屏幕上的半径
    if separate_sh:
        # 分离球谐函数处理模式
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,                 # 3D位置
            means2D = means2D,                 # 2D位置
            dc = dc,                           # 直流分量（基础颜色）
            shs = shs,                         # 球谐函数系数
            colors_precomp = colors_precomp,   # 预计算的颜色
            opacities = opacity,               # 不透明度
            scales = scales,                   # 缩放参数
            rotations = rotations,             # 旋转参数
            cov3D_precomp = cov3D_precomp)     # 预计算的3D协方差矩阵
    else:
        # 统一球谐函数处理模式
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,                 # 3D位置
            means2D = means2D,                 # 2D位置
            shs = shs,                         # 球谐函数系数
            colors_precomp = colors_precomp,   # 预计算的颜色
            opacities = opacity,               # 不透明度
            scales = scales,                   # 缩放参数
            rotations = rotations,             # 旋转参数
            cov3D_precomp = cov3D_precomp)     # 预计算的3D协方差矩阵
        
    # 对渲染图像应用曝光调整（仅在训练时使用）
    if use_trained_exp:
        # 从相机名称获取对应的曝光参数
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        # 应用曝光变换：先进行矩阵乘法（色彩校正），再加上偏移量
        # permute操作是为了调整张量维度以适应矩阵乘法
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # 那些被视锥剔除或半径为0的高斯点是不可见的
    # 它们将被排除在用于分裂标准的数值更新之外

    # 将渲染图像的像素值限制在[0, 1]范围内
    rendered_image = rendered_image.clamp(0, 1)

    # 构建输出字典，包含渲染结果和相关信息
    out = {
        "render": rendered_image,                        # 渲染的图像
        "viewspace_points": screenspace_points,          # 视图空间中的点坐标
        "visibility_filter" : (radii > 0).nonzero(),     # 可见性过滤器（半径>0的点）
        "radii": radii,                                  # 每个高斯点在屏幕上的半径
        "depth" : depth_image                            # 深度图像
        }
    
    return out
