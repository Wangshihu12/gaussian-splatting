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
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    """
    3D高斯点云渲染的主训练函数
    
    参数:
    - dataset: 数据集参数
    - opt: 优化器参数配置
    - pipe: 渲染管线参数
    - testing_iterations: 测试迭代列表
    - saving_iterations: 保存模型的迭代列表
    - checkpoint_iterations: 保存检查点的迭代列表
    - checkpoint: 检查点文件路径
    - debug_from: 开始调试的迭代次数
    """

    # 检查稀疏Adam优化器的可用性
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"尝试使用稀疏Adam优化器但未安装，请使用 pip install [3dgs_accel] 安装正确的光栅化器。")

    # 初始化训练参数
    first_iter = 0  # 起始迭代次数
    tb_writer = prepare_output_and_logger(dataset)   # 准备输出目录和TensorBoard日志记录器
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)    # 创建高斯模型
    scene = Scene(dataset, gaussians)   # 创建场景对象
    gaussians.training_setup(opt)    # 设置高斯模型的训练参数

    # 如果提供了检查点，则从检查点恢复模型
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # 设置背景颜色：白色背景或黑色背景
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 创建CUDA事件用于计时
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # 判断是否使用稀疏Adam优化器
    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 

    # 深度L1损失权重的指数衰减函数
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    # 初始化训练相机列表和索引
    viewpoint_stack = scene.getTrainCameras().copy()       # 复制训练相机列表
    viewpoint_indices = list(range(len(viewpoint_stack)))  # 相机索引列表

    # 初始化EMA（指数移动平均）损失记录
    ema_loss_for_log = 0.0       # 用于记录的EMA总损失
    ema_Ll1depth_for_log = 0.0   # 用于记录的EMA深度损失

    # 创建进度条
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # 主训练循环
    for iteration in range(first_iter, opt.iterations + 1):
        # 网络GUI连接处理（用于实时可视化）
        if network_gui.conn == None:
            network_gui.try_connect()  # 尝试连接GUI

        # 处理GUI交互
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                # 接收GUI命令和参数
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()

                # 如果提供了自定义相机，则进行渲染
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    # 将渲染结果转换为字节数组发送给GUI
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())

                # 发送图像数据给GUI
                network_gui.send(net_image_bytes, dataset.source_path)

                # 检查是否继续训练
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        # 开始计时
        iter_start.record()

        # 更新学习率
        gaussians.update_learning_rate(iteration)

        # 每1000次迭代增加球谐函数的阶数，直到最大阶数
        # 这是一个渐进式训练策略，从低阶开始逐步增加复杂度
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # 随机选择一个训练相机
        if not viewpoint_stack:
            # 如果相机列表为空，重新填充
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))

        # 随机选择一个相机索引
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)  # 取出选中的相机
        vind = viewpoint_indices.pop(rand_idx)         # 取出对应的索引

        # 执行渲染
        # 如果到达调试起始点，启用调试模式
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # 设置背景：如果启用随机背景则使用随机颜色，否则使用固定背景
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # 调用渲染函数获取渲染结果包
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # 如果存在alpha遮罩，应用到渲染图像上
        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # 计算损失函数
        gt_image = viewpoint_cam.original_image.cuda()  # 获取真实图像
        Ll1 = l1_loss(image, gt_image)                  # 计算L1损失（像素级差异）

        # 计算SSIM损失（结构相似性）
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        # 组合总损失：L1损失 + SSIM损失的加权组合
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # 深度正则化损失
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            # 如果启用深度损失且当前相机的深度可靠
            invDepth = render_pkg["depth"]                    # 渲染的逆深度图
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()  # 单目深度估计的逆深度图
            depth_mask = viewpoint_cam.depth_mask.cuda()      # 深度掩码

            # 计算深度L1损失（只在掩码区域内计算）
            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth  # 将深度损失加到总损失中
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        # 反向传播
        loss.backward()

        # 结束计时
        iter_end.record()

        # 无梯度上下文中处理日志记录和其他操作
        with torch.no_grad():
            # 更新EMA损失用于进度条显示
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            # 每10次迭代更新进度条
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # 记录训练日志和进行测试
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)

            # 在指定迭代保存模型
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # 密集化处理（在指定迭代范围内）
            if iteration < opt.densify_until_iter:
                # 跟踪图像空间中的最大半径用于剪枝
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                # 添加密集化统计信息
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # 在指定迭代范围内定期进行密集化和剪枝
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    # 设置大小阈值：如果迭代次数超过不透明度重置间隔则使用20，否则为None
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    # 执行密集化和剪枝操作
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                # 定期重置不透明度或在白色背景的特定迭代重置
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # 优化器步骤
            if iteration < opt.iterations:
                # 更新曝光参数优化器
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)

                # 根据优化器类型进行参数更新
                if use_sparse_adam:
                    # 稀疏Adam：只更新可见的高斯点
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    # 标准Adam：更新所有参数
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            # 在指定迭代保存检查点
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
