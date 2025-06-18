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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """
        Scene类的初始化方法：负责加载和管理3D场景数据
        
        Scene类是3D高斯渲染系统的核心组件，负责：
        1. 加载不同格式的3D重建数据（COLMAP、Blender等）
        2. 管理训练和测试相机
        3. 初始化高斯点云模型
        4. 处理多分辨率数据
        
        参数:
        :param args: ModelParams对象，包含模型相关的所有参数配置
        :param gaussians: GaussianModel对象，3D高斯点云模型
        :param load_iteration: 要加载的训练迭代次数
                            - None: 从头开始训练，使用初始点云
                            - -1: 自动加载最新的训练结果
                            - 具体数字: 加载指定迭代的模型
        :param shuffle: 是否随机打乱相机顺序，用于训练时的随机采样
        :param resolution_scales: 分辨率缩放因子列表，支持多分辨率训练
        """
        
        # ===== 基本属性初始化 =====
        self.model_path = args.model_path      # 模型保存路径
        self.loaded_iter = None                # 实际加载的迭代次数
        self.gaussians = gaussians             # 高斯模型对象引用

        # ===== 处理模型加载逻辑 =====
        if load_iteration:
            if load_iteration == -1:
                # 自动搜索最大迭代次数（最新的训练结果）
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                # 使用指定的迭代次数
                self.loaded_iter = load_iteration
            print("加载训练模型于迭代次数 {}".format(self.loaded_iter))

        # ===== 初始化相机容器 =====
        # 使用字典存储不同分辨率下的相机列表
        self.train_cameras = {}  # 训练相机：键为分辨率缩放因子，值为相机列表
        self.test_cameras = {}   # 测试相机：键为分辨率缩放因子，值为相机列表

        # ===== 场景类型检测和数据加载 =====
        # 根据数据目录结构自动识别场景类型
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            # 检测到"sparse"目录，说明是COLMAP格式的数据
            # COLMAP是常用的Structure-from-Motion工具，输出包含sparse重建结果
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path,      # 数据源路径
                args.images,           # 图像路径
                args.depths,           # 深度图路径（可选）
                args.eval,             # 是否为评估模式
                args.train_test_exp    # 训练测试曝光设置
            )
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            # 检测到"transforms_train.json"文件，说明是Blender格式的合成数据
            # Blender NeRF数据集通常包含精确的相机参数和渲染图像
            print("发现 transforms_train.json 文件, 假设为 Blender 数据集!")
            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path,      # 数据源路径
                args.white_background, # 是否使用白色背景
                args.depths,           # 深度图路径（可选）
                args.eval              # 是否为评估模式
            )
        else:
            # 如果都不匹配，抛出错误
            assert False, "无法识别场景类型!"

        # ===== 首次训练的初始化工作 =====
        if not self.loaded_iter:  # 如果不是加载已有模型，而是从头开始训练
            
            # ===== 复制初始点云文件 =====
            # 将原始点云文件复制到模型目录，作为训练的起始点
            with open(scene_info.ply_path, 'rb') as src_file, \
                open(os.path.join(self.model_path, "input.ply"), 'wb') as dest_file:
                dest_file.write(src_file.read())
            
            # ===== 生成相机参数JSON文件 =====
            # 将相机信息保存为JSON格式，便于查看和调试
            json_cams = []  # 存储JSON格式的相机信息
            camlist = []    # 临时存储所有相机
            
            # 收集所有相机（测试相机 + 训练相机）
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            
            # 将每个相机转换为JSON格式
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            
            # 保存相机参数到JSON文件
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        # ===== 随机打乱相机顺序 =====
        if shuffle:
            # 随机打乱训练和测试相机的顺序
            # 这对训练很重要，可以避免序列偏差，提高泛化能力
            # 注释中提到"Multi-res consistent"意味着在多分辨率训练中保持一致的随机顺序
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)

        # ===== 设置场景范围 =====
        # 从NeRF归一化信息中获取场景的空间范围
        # 这个半径用于确定高斯点的初始化范围和优化边界
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # ===== 创建多分辨率相机列表 =====
        # 为每个分辨率缩放因子创建对应的相机列表
        for resolution_scale in resolution_scales:
            print("加载训练相机")
            # 创建训练相机列表
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.train_cameras,    # 相机信息列表
                resolution_scale,            # 分辨率缩放因子
                args,                        # 模型参数
                scene_info.is_nerf_synthetic, # 是否为合成数据
                False                        # 不是测试数据集
            )
            
            print("加载测试相机")
            # 创建测试相机列表
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.test_cameras,     # 相机信息列表
                resolution_scale,            # 分辨率缩放因子
                args,                        # 模型参数
                scene_info.is_nerf_synthetic, # 是否为合成数据
                True                         # 是测试数据集
            )

        # ===== 初始化高斯模型 =====
        if self.loaded_iter:
            # 如果加载已有模型，从PLY文件中加载训练好的高斯点
            self.gaussians.load_ply(
                os.path.join(self.model_path, "point_cloud", 
                            f"iteration_{self.loaded_iter}", "point_cloud.ply"),
                args.train_test_exp
            )
        else:
            # 如果从头开始训练，从初始点云创建高斯点
            self.gaussians.create_from_pcd(
                scene_info.point_cloud,      # 初始点云数据
                scene_info.train_cameras,    # 训练相机信息
                self.cameras_extent          # 场景范围
            )

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }

        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
