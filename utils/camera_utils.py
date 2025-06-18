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

from scene.cameras import Camera
import numpy as np
from utils.graphics_utils import fov2focal
from PIL import Image
import cv2

WARNED = False

def loadCam(args, id, cam_info, resolution_scale, is_nerf_synthetic, is_test_dataset):
    """
    加载相机数据函数：从文件系统加载图像和深度信息，创建Camera对象
    
    参数:
    - args: 命令行参数对象，包含分辨率设置等配置
    - id: 相机的唯一标识符
    - cam_info: 相机信息对象，包含图像路径、深度路径、相机参数等
    - resolution_scale: 分辨率缩放因子
    - is_nerf_synthetic: 是否为NeRF合成数据集
    - is_test_dataset: 是否为测试数据集
    
    返回:
    - Camera对象：包含所有相机数据和参数
    """
    
    # ===== 加载RGB图像 =====
    image = Image.open(cam_info.image_path)  # 使用PIL加载图像文件
    
    # ===== 加载深度图（如果存在） =====
    if cam_info.depth_path != "":  # 检查是否提供了深度图路径
        try:
            # 根据数据集类型使用不同的深度值归一化方式
            if is_nerf_synthetic:
                # NeRF合成数据集：深度值除以512进行归一化
                # 这是因为合成数据集通常使用特定的深度范围编码
                invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / 512
            else:
                # 真实数据集：深度值除以2^16进行归一化
                # 这通常对应16位深度图的标准归一化方式
                invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / float(2**16)

        # ===== 异常处理 =====
        except FileNotFoundError:
            # 文件不存在错误
            print(f"错误: 深度图文件路径 '{cam_info.depth_path}' 未找到。")
            raise
        except IOError:
            # 文件IO错误（文件损坏或格式不支持）
            print(f"错误: 无法打开图像文件 '{cam_info.depth_path}'。它可能已损坏或不支持的格式。")
            raise
        except Exception as e:
            # 其他未预期的错误
            print(f"在尝试读取深度图时发生意外错误: {e}")
            raise
    else:
        # 如果没有提供深度图路径，设置为None
        invdepthmap = None
        
    # ===== 获取原始图像尺寸 =====
    orig_w, orig_h = image.size  # 获取图像的原始宽度和高度
    
    # ===== 计算目标分辨率 =====
    if args.resolution in [1, 2, 4, 8]:
        # 如果分辨率参数是预定义的缩放因子（1, 2, 4, 8）
        # 计算目标分辨率：原始尺寸 / (分辨率缩放因子 * 用户指定的分辨率因子)
        resolution = (
            round(orig_w / (resolution_scale * args.resolution)), 
            round(orig_h / (resolution_scale * args.resolution))
        )
    else:  
        # 如果分辨率参数是具体的像素值或特殊值
        if args.resolution == -1:
            # 自动分辨率模式
            if orig_w > 1600:
                # 如果图像宽度超过1600像素，自动缩放到1600像素
                global WARNED  # 全局警告标志，避免重复警告
                if not WARNED:
                    print("[ INFO ] 遇到较大的输入图像 (>1.6K像素宽度), 自动缩放到1.6K。\n "
                        "如果不需要，请显式指定 '--resolution/-r' 为1")
                    WARNED = True
                global_down = orig_w / 1600  # 计算缩放比例
            else:
                global_down = 1  # 不需要缩放
        else:
            # 用户指定了具体的目标宽度
            global_down = orig_w / args.resolution
    
        # 计算最终的缩放比例：全局缩放 * 分辨率缩放
        scale = float(global_down) * float(resolution_scale)
        # 计算最终分辨率
        resolution = (int(orig_w / scale), int(orig_h / scale))

    # ===== 创建并返回Camera对象 =====
    return Camera(
        resolution,                           # 目标分辨率
        colmap_id=cam_info.uid,              # COLMAP相机ID
        R=cam_info.R,                        # 旋转矩阵
        T=cam_info.T,                        # 平移向量
        FoVx=cam_info.FovX,                  # X方向视场角
        FoVy=cam_info.FovY,                  # Y方向视场角
        depth_params=cam_info.depth_params,  # 深度参数
        image=image,                         # RGB图像
        invdepthmap=invdepthmap,            # 逆深度图
        image_name=cam_info.image_name,      # 图像名称
        uid=id,                             # 唯一标识符
        data_device=args.data_device,        # 数据存储设备（CPU/GPU）
        train_test_exp=args.train_test_exp,  # 训练测试曝光标志
        is_test_dataset=is_test_dataset,     # 是否为测试数据集
        is_test_view=cam_info.is_test        # 是否为测试视角
    )

def cameraList_from_camInfos(cam_infos, resolution_scale, args, is_nerf_synthetic, is_test_dataset):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale, is_nerf_synthetic, is_test_dataset))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry