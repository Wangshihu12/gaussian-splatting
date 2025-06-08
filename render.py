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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    """
    渲染单个数据集（训练集或测试集）的函数
    
    参数:
    - model_path: 模型保存路径
    - name: 数据集名称（"train"或"test"）
    - iteration: 当前迭代次数
    - views: 相机视角列表
    - gaussians: 高斯模型对象
    - pipeline: 渲染管线参数
    - background: 背景颜色张量
    - train_test_exp: 训练测试实验标志
    - separate_sh: 是否分离球谐函数处理
    """

    # 构建渲染结果保存路径
    # 格式：model_path/数据集名称/ours_迭代次数/renders/
    # 例如：output/train/ours_30000/renders/
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")

    # 构建ground truth（真实图像）保存路径
    # 格式：model_path/数据集名称/ours_迭代次数/gt/
    # 例如：output/train/ours_30000/gt/
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    # 创建渲染结果目录，exist_ok=True表示如果目录已存在则不报错
    makedirs(render_path, exist_ok=True)

    # 创建ground truth目录
    makedirs(gts_path, exist_ok=True)

    # 遍历所有相机视角进行渲染
    # enumerate获取索引和视角对象，tqdm显示进度条
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # 调用render函数进行实际渲染
        # 返回字典中的"render"键包含渲染结果
        # use_trained_exp和separate_sh控制渲染的具体行为
        rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]

        # 获取当前视角的原始图像（ground truth）
        # [0:3, :, :] 表示取前3个通道（RGB），排除alpha通道
        gt = view.original_image[0:3, :, :]

        # 如果启用了训练测试实验模式
        # 注意：这里应该使用参数train_test_exp而不是全局变量args.train_test_exp
        if args.train_test_exp:
            # 只保留图像的右半部分
            # rendering.shape[-1] // 2 计算图像宽度的一半
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        # 保存渲染结果图像
        # 文件名格式：00000.png, 00001.png, ..., 使用5位数字编号
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

        # 保存ground truth图像，使用相同的文件名格式
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool):
    """
    渲染数据集的函数
    
    参数:
    - dataset: 模型参数，包含数据集相关配置
    - iteration: 迭代次数，用于加载特定迭代的模型
    - pipeline: 渲染管线参数
    - skip_train: 是否跳过训练集渲染
    - skip_test: 是否跳过测试集渲染  
    - separate_sh: 是否分离球谐函数处理
    """

    # 使用torch.no_grad()禁用梯度计算，因为这是推理阶段，不需要计算梯度
    # 这样可以节省内存并提高计算速度
    with torch.no_grad():
        # 创建高斯模型实例，sh_degree是球谐函数的阶数
        # 球谐函数用于表示高斯椭球的颜色信息
        gaussians = GaussianModel(dataset.sh_degree)

        # 创建场景对象，加载指定迭代次数的模型权重
        # shuffle=False确保相机顺序的一致性，便于结果比较
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        # 根据数据集设置确定背景颜色
        # 如果white_background为True，使用白色背景[1,1,1]
        # 否则使用黑色背景[0,0,0]
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]

        # 将背景颜色转换为CUDA张量，用于GPU计算
        # dtype=torch.float32确保数据类型一致性
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # 如果不跳过训练集渲染
        if not skip_train:
             # 渲染训练集视角
             # 参数包括：模型路径、数据集名称、迭代次数、训练相机列表、高斯模型、
             # 管线参数、背景颜色、训练测试实验设置、球谐函数分离设置
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

        # 如果不跳过测试集渲染
        if not skip_test:
             # 渲染测试集视角，参数与训练集类似
             # 但使用测试相机列表getTestCameras()
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)