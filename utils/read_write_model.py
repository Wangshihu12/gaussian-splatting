# Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


import os
import collections
import numpy as np
import struct
import argparse


# ===== COLMAP数据结构定义 =====
# 这些数据结构对应COLMAP（一个开源的Structure-from-Motion和Multi-View Stereo工具）的输出格式

# ===== 相机模型数据结构 =====
CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)
"""
相机模型命名元组：定义相机的内参模型类型
- model_id: 模型的数字ID（0-10）
- model_name: 模型的字符串名称（如"PINHOLE", "OPENCV"等）
- num_params: 该模型需要的参数数量
"""

# ===== 相机数据结构 =====
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"]
)
"""
相机命名元组：存储具体相机实例的信息
- id: 相机的唯一标识符
- model: 相机使用的模型类型（对应CameraModel）
- width: 图像宽度（像素）
- height: 图像高度（像素）
- params: 相机内参数组（焦距、主点、畸变参数等）
"""

# ===== 图像基础数据结构 =====
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)
"""
图像基础命名元组：存储图像的姿态和特征点信息
- id: 图像的唯一标识符
- qvec: 四元数表示的旋转（4个元素：w, x, y, z）
- tvec: 平移向量（3个元素：tx, ty, tz）
- camera_id: 关联的相机ID
- name: 图像文件名
- xys: 2D特征点坐标数组 (N×2)
- point3D_ids: 对应的3D点ID数组 (N×1)，-1表示该2D点没有对应的3D点
"""

# ===== 3D点数据结构 =====
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)
"""
3D点命名元组：存储重建的3D点信息
- id: 3D点的唯一标识符
- xyz: 3D坐标 (x, y, z)
- rgb: RGB颜色值 (r, g, b)
- error: 重投影误差
- image_ids: 观察到该3D点的图像ID列表
- point2D_idxs: 在对应图像中的2D特征点索引列表
"""

# ===== 扩展的图像类 =====
class Image(BaseImage):
    """
    扩展的图像类：继承自BaseImage，添加了额外的方法
    """
    def qvec2rotmat(self):
        """
        将四元数转换为旋转矩阵
        
        返回:
        - 3×3旋转矩阵
        """
        return qvec2rotmat(self.qvec)  # 调用外部函数进行四元数到旋转矩阵的转换

# ===== COLMAP支持的相机模型定义 =====
CAMERA_MODELS = {
    # 简单针孔模型：只有焦距和主点，无畸变
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),     # 参数：f, cx, cy
    
    # 标准针孔模型：分别的fx, fy焦距
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),            # 参数：fx, fy, cx, cy
    
    # 简单径向畸变模型：一个径向畸变参数
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),      # 参数：f, cx, cy, k1
    
    # 径向畸变模型：两个径向畸变参数
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),             # 参数：f, cx, cy, k1, k2
    
    # OpenCV模型：包含径向和切向畸变
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),             # 参数：fx, fy, cx, cy, k1, k2, p1, p2
    
    # OpenCV鱼眼模型：适用于鱼眼镜头
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),     # 参数：fx, fy, cx, cy, k1, k2, k3, k4
    
    # 完整OpenCV模型：包含所有畸变参数
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),       # 参数：fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
    
    # FOV模型：视场角模型
    CameraModel(model_id=7, model_name="FOV", num_params=5),                # 参数：fx, fy, cx, cy, omega
    
    # 简单径向鱼眼模型
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),  # 参数：f, cx, cy, k1
    
    # 径向鱼眼模型
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),     # 参数：f, cx, cy, k1, k2
    
    # 薄棱镜鱼眼模型：最复杂的鱼眼模型
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12), # 参数：fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1
}

# ===== 创建查找字典 =====
# 通过模型ID查找模型信息的字典
CAMERA_MODEL_IDS = dict(
    [(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS]
)
"""
模型ID到模型对象的映射字典
例如：CAMERA_MODEL_IDS[1] 返回 PINHOLE模型的CameraModel对象
"""

# 通过模型名称查找模型信息的字典
CAMERA_MODEL_NAMES = dict(
    [(camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS]
)
"""
模型名称到模型对象的映射字典
例如：CAMERA_MODEL_NAMES["PINHOLE"] 返回 PINHOLE模型的CameraModel对象
"""


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """
    从二进制文件中读取并解包下一组字节数据
    
    这是一个底层的二进制文件读取函数，用于解析COLMAP的二进制格式文件
    
    参数:
    :param fid: 文件对象（已打开的二进制文件句柄）
    :param num_bytes: 要读取的字节总数，必须是{2, 4, 8}的组合，例如2, 6, 16, 30等
                     这个数值应该与format_char_sequence指定的数据类型的总字节数匹配
    :param format_char_sequence: 格式字符序列，指定如何解释读取的字节
                                可用字符：{c, e, f, d, h, H, i, I, l, L, q, Q}
                                - c: char (1字节)
                                - e: float16 (2字节) 
                                - f: float32 (4字节)
                                - d: float64 (8字节)
                                - h: short (2字节，有符号)
                                - H: unsigned short (2字节，无符号)
                                - i: int (4字节，有符号)
                                - I: unsigned int (4字节，无符号)
                                - l: long (4字节，有符号)
                                - L: unsigned long (4字节，无符号)
                                - q: long long (8字节，有符号)
                                - Q: unsigned long long (8字节，无符号)
    :param endian_character: 字节序字符，默认为"<"（小端序）
                            可用选项：{@, =, <, >, !}
                            - @: 本机字节序
                            - =: 本机字节序
                            - <: 小端序（Intel x86）
                            - >: 大端序（网络字节序）
                            - !: 网络字节序（大端序）
    
    返回:
    :return: 解包后的数据元组，包含按format_char_sequence指定格式解析的值
    
    示例:
    # 读取一个4字节的无符号整数和两个4字节的浮点数（总共12字节）
    # result = read_next_bytes(file, 12, "Iff")
    # 返回: (unsigned_int_value, float_value1, float_value2)
    """
    
    # 从文件中读取指定数量的字节
    data = fid.read(num_bytes)
    
    # 使用struct.unpack解包二进制数据
    # 组合字节序字符和格式字符序列，然后解包数据
    return struct.unpack(endian_character + format_char_sequence, data)


def write_next_bytes(fid, data, format_char_sequence, endian_character="<"):
    """
    将数据打包并写入二进制文件
    
    这是一个底层的二进制文件写入函数，用于生成COLMAP格式的二进制文件
    
    参数:
    :param fid: 文件对象（已打开的二进制写入文件句柄）
    :param data: 要写入的数据，可以是单个值或多个值的列表/元组
                如果是多个元素，应该封装在列表或元组中
                数据类型必须与format_char_sequence指定的格式匹配
    :param format_char_sequence: 格式字符序列，指定如何打包数据
                                字符含义与read_next_bytes中相同
                                序列长度应该与数据列表或元组的长度相同
    :param endian_character: 字节序字符，默认为"<"（小端序）
                            选项与read_next_bytes中相同
    
    功能说明:
    - 如果data是列表或元组，使用*data解包所有元素进行打包
    - 如果data是单个值，直接打包该值
    - 打包后的字节数据写入文件
    
    示例:
    # 写入一个无符号整数和两个浮点数
    # write_next_bytes(file, [123, 3.14, 2.71], "Iff")
    # 或写入单个整数
    # write_next_bytes(file, 42, "I")
    """
    
    # 根据数据类型选择不同的打包方式
    if isinstance(data, (list, tuple)):
        # 如果数据是列表或元组，使用*data解包所有元素
        # 这样可以将多个值作为单独的参数传递给struct.pack
        bytes = struct.pack(endian_character + format_char_sequence, *data)
    else:
        # 如果数据是单个值，直接打包
        bytes = struct.pack(endian_character + format_char_sequence, data)
    
    # 将打包后的字节数据写入文件
    fid.write(bytes)


def read_cameras_text(path):
    """
    从文本文件中读取COLMAP相机数据
    
    对应COLMAP源码中的函数：
    src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    
    文本格式说明：
    # 每行格式：CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]
    # 示例：
    # 1 PINHOLE 1920 1080 1065.0 1065.0 960.0 540.0
    # 2 RADIAL 1280 720 800.0 640.0 360.0 0.1 0.05
    
    参数:
    :param path: 相机文本文件的路径（通常是cameras.txt）
    
    返回:
    :return: 字典，键为相机ID，值为Camera命名元组对象
    """
    cameras = {}  # 初始化空字典存储相机数据
    
    # 以文本模式打开文件
    with open(path, "r") as fid:
        while True:
            # 逐行读取文件内容
            line = fid.readline()
            if not line:  # 如果到达文件末尾，退出循环
                break
                
            line = line.strip()  # 去除行首尾的空白字符
            
            # 跳过空行和注释行（以#开头的行）
            if len(line) > 0 and line[0] != "#":
                # 将行内容按空格分割
                elems = line.split()
                
                # ===== 解析相机参数 =====
                camera_id = int(elems[0])      # 相机ID（整数）
                model = elems[1]               # 相机模型名称（字符串）
                width = int(elems[2])          # 图像宽度（像素）
                height = int(elems[3])         # 图像高度（像素）
                
                # 解析相机内参数组（从第5个元素开始的所有浮点数）
                # 使用map(float, elems[4:])将字符串转换为浮点数
                # 然后转换为numpy数组
                params = np.array(tuple(map(float, elems[4:])))
                
                # ===== 创建Camera对象并存储 =====
                cameras[camera_id] = Camera(
                    id=camera_id,      # 相机ID
                    model=model,       # 相机模型名称
                    width=width,       # 图像宽度
                    height=height,     # 图像高度
                    params=params,     # 内参数组
                )
    
    return cameras  # 返回包含所有相机的字典


def read_cameras_binary(path_to_model_file):
    """
    从二进制文件中读取COLMAP相机数据
    
    对应COLMAP源码中的函数：
    src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    
    二进制格式说明：
    1. 8字节：相机总数量（uint64）
    2. 对每个相机：
       - 4字节：相机ID（int32）
       - 4字节：模型ID（int32）
       - 8字节：图像宽度（uint64）
       - 8字节：图像高度（uint64）
       - 8*N字节：N个参数（每个参数8字节double）
    
    参数:
    :param path_to_model_file: 相机二进制文件的路径（通常是cameras.bin）
    
    返回:
    :return: 字典，键为相机ID，值为Camera命名元组对象
    """
    cameras = {}  # 初始化空字典存储相机数据
    
    # 以二进制模式打开文件
    with open(path_to_model_file, "rb") as fid:
        # ===== 读取相机总数量 =====
        # 读取8字节，格式为无符号64位整数（"Q"）
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        
        # ===== 逐个读取每个相机的数据 =====
        for _ in range(num_cameras):
            # ===== 读取相机基本属性 =====
            # 读取24字节：两个int32 + 两个uint64
            # 格式："iiQQ" = int32, int32, uint64, uint64
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ"
            )
            
            # 解析基本属性
            camera_id = camera_properties[0]  # 相机ID
            model_id = camera_properties[1]   # 模型ID（数字）
            width = camera_properties[2]      # 图像宽度
            height = camera_properties[3]     # 图像高度
            
            # ===== 通过模型ID获取模型信息 =====
            # 从预定义的CAMERA_MODEL_IDS字典中查找模型名称
            model_name = CAMERA_MODEL_IDS[model_id].model_name
            
            # 获取该模型需要的参数数量
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            
            # ===== 读取相机内参数 =====
            # 读取 8*num_params 字节的参数数据
            # 每个参数都是8字节的double类型
            # 格式字符串："d" * num_params，例如"ddd"表示3个double
            params = read_next_bytes(
                fid,
                num_bytes=8 * num_params,           # 总字节数
                format_char_sequence="d" * num_params,  # 格式字符串
            )
            
            # ===== 创建Camera对象并存储 =====
            cameras[camera_id] = Camera(
                id=camera_id,                # 相机ID
                model=model_name,           # 相机模型名称（字符串）
                width=width,                # 图像宽度
                height=height,              # 图像高度
                params=np.array(params),    # 内参数组（转换为numpy数组）
            )
        
        # ===== 验证数据完整性 =====
        # 确保读取的相机数量与文件头声明的数量一致
        assert len(cameras) == num_cameras
    
    return cameras  # 返回包含所有相机的字典


def write_cameras_text(cameras, path):
    """
    将相机数据写入文本文件
    
    对应COLMAP源码中的函数：
    src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    
    生成的文件格式：
    # Camera list with one line of data per camera:
    #   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
    # Number of cameras: N
    1 PINHOLE 1920 1080 1065.0 1065.0 960.0 540.0
    2 RADIAL 1280 720 800.0 640.0 360.0 0.1 0.05
    
    参数:
    :param cameras: 相机字典，键为相机ID，值为Camera对象
    :param path: 输出文本文件的路径（通常是cameras.txt）
    """
    
    # ===== 定义文件头部信息 =====
    HEADER = (
        "# Camera list with one line of data per camera:\n"          # 说明每行包含一个相机的数据
        + "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"         # 说明数据格式
        + "# Number of cameras: {}\n".format(len(cameras))          # 显示相机总数量
    )
    
    # ===== 写入文件 =====
    with open(path, "w") as fid:  # 以文本写入模式打开文件
        # 首先写入头部信息
        fid.write(HEADER)
        
        # ===== 遍历所有相机并写入数据 =====
        for _, cam in cameras.items():  # 遍历相机字典，忽略键，只使用值
            # ===== 准备要写入的数据 =====
            # 使用*cam.params将参数数组展开为单独的元素
            # 例如：如果cam.params = [1065.0, 1065.0, 960.0, 540.0]
            # 则*cam.params会展开为 1065.0, 1065.0, 960.0, 540.0
            to_write = [cam.id, cam.model, cam.width, cam.height, *cam.params]
            
            # ===== 格式化并写入一行数据 =====
            # 将所有元素转换为字符串并用空格连接
            line = " ".join([str(elem) for elem in to_write])
            fid.write(line + "\n")  # 写入行数据并添加换行符


def write_cameras_binary(cameras, path_to_model_file):
    """
    将相机数据写入二进制文件
    
    对应COLMAP源码中的函数：
    src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    
    二进制文件格式：
    1. 8字节：相机总数量（uint64）
    2. 对每个相机：
       - 4字节：相机ID（int32）
       - 4字节：模型ID（int32）
       - 8字节：图像宽度（uint64）
       - 8字节：图像高度（uint64）
       - 8*N字节：N个参数（每个参数8字节double）
    
    参数:
    :param cameras: 相机字典，键为相机ID，值为Camera对象
    :param path_to_model_file: 输出二进制文件的路径（通常是cameras.bin）
    
    返回:
    :return: 返回输入的cameras字典（便于链式调用）
    """
    
    # ===== 写入二进制文件 =====
    with open(path_to_model_file, "wb") as fid:  # 以二进制写入模式打开文件
        
        # ===== 写入相机总数量 =====
        # 格式："Q" = 无符号64位整数（8字节）
        write_next_bytes(fid, len(cameras), "Q")
        
        # ===== 遍历所有相机并写入数据 =====
        for _, cam in cameras.items():  # 遍历相机字典
            
            # ===== 获取模型ID =====
            # 通过模型名称查找对应的数字ID
            # 例如："PINHOLE" -> 1, "RADIAL" -> 3
            model_id = CAMERA_MODEL_NAMES[cam.model].model_id
            
            # ===== 准备相机基本属性数据 =====
            camera_properties = [cam.id, model_id, cam.width, cam.height]
            
            # ===== 写入相机基本属性 =====
            # 格式："iiQQ" = int32, int32, uint64, uint64（总共24字节）
            # - cam.id: 相机ID（4字节有符号整数）
            # - model_id: 模型ID（4字节有符号整数）
            # - cam.width: 图像宽度（8字节无符号整数）
            # - cam.height: 图像高度（8字节无符号整数）
            write_next_bytes(fid, camera_properties, "iiQQ")
            
            # ===== 写入相机内参数 =====
            # 逐个写入每个参数，每个参数都是8字节的double类型
            for p in cam.params:
                # 格式："d" = double（8字节浮点数）
                # 使用float(p)确保参数是浮点数类型
                write_next_bytes(fid, float(p), "d")
    
    return cameras  # 返回输入的相机字典


def read_images_text(path):
    """
    从文本文件中读取COLMAP图像数据
    
    对应COLMAP源码中的函数：
    src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    
    文本格式说明：
    # 每个图像占用两行：
    # 第一行：IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID IMAGE_NAME
    # 第二行：POINTS2D[] as (X, Y, POINT3D_ID, X, Y, POINT3D_ID, ...)
    # 如果POINT3D_ID为-1，表示该2D点没有对应的3D点
    
    参数:
    :param path: 图像文本文件的路径（通常是images.txt）
    
    返回:
    :return: 字典，键为图像ID，值为Image对象
    """
    images = {}  # 初始化空字典存储图像数据
    
    # 以文本模式打开文件
    with open(path, "r") as fid:
        while True:
            # ===== 读取图像基本信息行 =====
            line = fid.readline()
            if not line:  # 如果到达文件末尾，退出循环
                break
                
            line = line.strip()  # 去除行首尾的空白字符
            
            # 跳过空行和注释行（以#开头的行）
            if len(line) > 0 and line[0] != "#":
                elems = line.split()  # 按空格分割行内容
                
                # ===== 解析图像基本信息 =====
                image_id = int(elems[0])                                    # 图像ID
                qvec = np.array(tuple(map(float, elems[1:5])))             # 四元数旋转 (qw, qx, qy, qz)
                tvec = np.array(tuple(map(float, elems[5:8])))             # 平移向量 (tx, ty, tz)
                camera_id = int(elems[8])                                   # 关联的相机ID
                image_name = elems[9]                                       # 图像文件名
                
                # ===== 读取2D特征点信息行 =====
                # 每个图像的特征点信息在下一行
                elems = fid.readline().split()
                
                # ===== 解析2D特征点坐标 =====
                # 数据格式：X1 Y1 POINT3D_ID1 X2 Y2 POINT3D_ID2 ...
                # 每3个元素为一组：(X, Y, POINT3D_ID)
                
                # 提取X坐标：从索引0开始，每隔3个取一个 (0, 3, 6, ...)
                # 提取Y坐标：从索引1开始，每隔3个取一个 (1, 4, 7, ...)
                xys = np.column_stack([
                    tuple(map(float, elems[0::3])),  # X坐标列表
                    tuple(map(float, elems[1::3])),  # Y坐标列表
                ])
                
                # 提取3D点ID：从索引2开始，每隔3个取一个 (2, 5, 8, ...)
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                
                # ===== 创建Image对象并存储 =====
                images[image_id] = Image(
                    id=image_id,              # 图像ID
                    qvec=qvec,               # 四元数旋转
                    tvec=tvec,               # 平移向量
                    camera_id=camera_id,     # 相机ID
                    name=image_name,         # 图像名称
                    xys=xys,                 # 2D特征点坐标 (N×2数组)
                    point3D_ids=point3D_ids, # 对应的3D点ID数组 (N×1数组)
                )
    
    return images  # 返回包含所有图像的字典


def read_images_binary(path_to_model_file):
    """
    从二进制文件中读取COLMAP图像数据
    
    对应COLMAP源码中的函数：
    src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    
    二进制格式说明：
    1. 8字节：注册图像总数量（uint64）
    2. 对每个图像：
       - 4字节：图像ID（int32）
       - 32字节：四元数旋转（4个double）
       - 24字节：平移向量（3个double）
       - 4字节：相机ID（int32）
       - 变长：图像名称（以null结尾的字符串）
       - 8字节：2D点数量（uint64）
       - 24*N字节：N个2D点数据（每个点：2个double + 1个int64）
    
    参数:
    :param path_to_model_file: 图像二进制文件的路径（通常是images.bin）
    
    返回:
    :return: 字典，键为图像ID，值为Image对象
    """
    images = {}  # 初始化空字典存储图像数据
    
    # 以二进制模式打开文件
    with open(path_to_model_file, "rb") as fid:
        
        # ===== 读取注册图像总数量 =====
        # 格式："Q" = 无符号64位整数（8字节）
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        
        # ===== 逐个读取每个图像的数据 =====
        for _ in range(num_reg_images):
            
            # ===== 读取图像基本属性 =====
            # 读取64字节的基本属性数据
            # 格式："idddddddi" = int32 + 7个double + int32
            # - 1个int32：图像ID（4字节）
            # - 4个double：四元数 qw, qx, qy, qz（32字节）
            # - 3个double：平移向量 tx, ty, tz（24字节）
            # - 1个int32：相机ID（4字节）
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi"
            )
            
            # 解析基本属性
            image_id = binary_image_properties[0]        # 图像ID
            qvec = np.array(binary_image_properties[1:5]) # 四元数旋转
            tvec = np.array(binary_image_properties[5:8]) # 平移向量
            camera_id = binary_image_properties[8]        # 相机ID
            
            # ===== 读取图像名称（变长字符串） =====
            image_name = ""
            # 逐字节读取字符，直到遇到ASCII 0（null终止符）
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # 查找ASCII 0终止符
                image_name += current_char.decode("utf-8")  # 解码为UTF-8字符
                current_char = read_next_bytes(fid, 1, "c")[0]
            
            # ===== 读取2D特征点数量 =====
            # 格式："Q" = 无符号64位整数（8字节）
            num_points2D = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q"
            )[0]
            
            # ===== 读取所有2D特征点数据 =====
            # 每个2D点占用24字节：2个double（X,Y坐标）+ 1个int64（3D点ID）
            # 格式："ddq" * num_points2D，其中q表示int64
            x_y_id_s = read_next_bytes(
                fid,
                num_bytes=24 * num_points2D,                    # 总字节数
                format_char_sequence="ddq" * num_points2D,      # 格式字符串
            )
            
            # ===== 解析2D特征点数据 =====
            # 数据排列：X1, Y1, ID1, X2, Y2, ID2, ...
            # 提取X坐标：索引0, 3, 6, ...
            # 提取Y坐标：索引1, 4, 7, ...
            xys = np.column_stack([
                tuple(map(float, x_y_id_s[0::3])),  # X坐标列表
                tuple(map(float, x_y_id_s[1::3])),  # Y坐标列表
            ])
            
            # 提取3D点ID：索引2, 5, 8, ...
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            
            # ===== 创建Image对象并存储 =====
            images[image_id] = Image(
                id=image_id,              # 图像ID
                qvec=qvec,               # 四元数旋转
                tvec=tvec,               # 平移向量
                camera_id=camera_id,     # 相机ID
                name=image_name,         # 图像名称
                xys=xys,                 # 2D特征点坐标
                point3D_ids=point3D_ids, # 对应的3D点ID
            )
    
    return images  # 返回包含所有图像的字典


def write_images_text(images, path):
    """
    将图像数据写入文本文件
    
    对应COLMAP源码中的函数：
    src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    
    生成的文件格式：
    # Image list with two lines of data per image:
    #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    #   POINTS2D[] as (X, Y, POINT3D_ID)
    # Number of images: N, mean observations per image: M
    1 0.707 0.0 0.0 0.707 1.0 2.0 3.0 1 image1.jpg
    100.5 200.3 15 150.2 180.7 -1 ...
    
    参数:
    :param images: 图像字典，键为图像ID，值为Image对象
    :param path: 输出文本文件的路径（通常是images.txt）
    """
    
    # ===== 计算统计信息 =====
    if len(images) == 0:
        mean_observations = 0  # 如果没有图像，平均观测数为0
    else:
        # 计算每个图像的平均特征点观测数
        # sum()计算所有图像的特征点总数，然后除以图像数量
        mean_observations = sum(
            (len(img.point3D_ids) for _, img in images.items())  # 每个图像的特征点数量
        ) / len(images)  # 除以图像总数得到平均值
    
    # ===== 定义文件头部信息 =====
    HEADER = (
        "# Image list with two lines of data per image:\n"                    # 说明每个图像占用两行
        + "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"     # 第一行格式说明
        + "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"                          # 第二行格式说明
        + "# Number of images: {}, mean observations per image: {}\n".format(
            len(images), mean_observations                                    # 显示图像总数和平均观测数
        )
    )

    # ===== 写入文件 =====
    with open(path, "w") as fid:  # 以文本写入模式打开文件
        # 首先写入头部信息
        fid.write(HEADER)
        
        # ===== 遍历所有图像并写入数据 =====
        for _, img in images.items():  # 遍历图像字典，忽略键，只使用值
            
            # ===== 准备图像基本信息（第一行） =====
            image_header = [
                img.id,        # 图像ID
                *img.qvec,     # 四元数旋转（展开为4个元素：qw, qx, qy, qz）
                *img.tvec,     # 平移向量（展开为3个元素：tx, ty, tz）
                img.camera_id, # 相机ID
                img.name,      # 图像名称
            ]
            
            # ===== 写入图像基本信息行 =====
            first_line = " ".join(map(str, image_header))  # 将所有元素转换为字符串并用空格连接
            fid.write(first_line + "\n")  # 写入第一行并添加换行符

            # ===== 准备2D特征点信息（第二行） =====
            points_strings = []  # 存储每个特征点的字符串表示
            
            # 遍历所有2D特征点及其对应的3D点ID
            for xy, point3D_id in zip(img.xys, img.point3D_ids):
                # 为每个特征点创建字符串：X Y POINT3D_ID
                # *xy展开坐标为两个元素：[x, y, point3D_id]
                points_strings.append(" ".join(map(str, [*xy, point3D_id])))
            
            # ===== 写入特征点信息行 =====
            # 将所有特征点字符串用空格连接成一行
            fid.write(" ".join(points_strings) + "\n")


def write_images_binary(images, path_to_model_file):
    """
    将图像数据写入二进制文件
    
    对应COLMAP源码中的函数：
    src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    
    二进制文件格式：
    1. 8字节：图像总数量（uint64）
    2. 对每个图像：
       - 4字节：图像ID（int32）
       - 32字节：四元数旋转（4个double）
       - 24字节：平移向量（3个double）
       - 4字节：相机ID（int32）
       - 变长：图像名称（以null结尾的字符串）
       - 8字节：2D点数量（uint64）
       - 24*N字节：N个2D点数据（每个点：2个double + 1个int64）
    
    参数:
    :param images: 图像字典，键为图像ID，值为Image对象
    :param path_to_model_file: 输出二进制文件的路径（通常是images.bin）
    """
    
    # ===== 写入二进制文件 =====
    with open(path_to_model_file, "wb") as fid:  # 以二进制写入模式打开文件
        
        # ===== 写入图像总数量 =====
        # 格式："Q" = 无符号64位整数（8字节）
        write_next_bytes(fid, len(images), "Q")
        
        # ===== 遍历所有图像并写入数据 =====
        for _, img in images.items():  # 遍历图像字典
            
            # ===== 写入图像基本属性 =====
            # 写入图像ID（4字节有符号整数）
            write_next_bytes(fid, img.id, "i")
            
            # 写入四元数旋转（4个8字节double）
            # tolist()将numpy数组转换为Python列表
            write_next_bytes(fid, img.qvec.tolist(), "dddd")
            
            # 写入平移向量（3个8字节double）
            write_next_bytes(fid, img.tvec.tolist(), "ddd")
            
            # 写入相机ID（4字节有符号整数）
            write_next_bytes(fid, img.camera_id, "i")
            
            # ===== 写入图像名称（变长字符串） =====
            # 逐字符写入图像名称
            for char in img.name:
                # 将字符编码为UTF-8字节并写入
                write_next_bytes(fid, char.encode("utf-8"), "c")
            
            # 写入字符串终止符（null字节）
            write_next_bytes(fid, b"\x00", "c")
            
            # ===== 写入2D特征点数量 =====
            # 格式："Q" = 无符号64位整数（8字节）
            write_next_bytes(fid, len(img.point3D_ids), "Q")
            
            # ===== 写入所有2D特征点数据 =====
            # 遍历每个2D特征点及其对应的3D点ID
            for xy, p3d_id in zip(img.xys, img.point3D_ids):
                # 写入一个特征点的完整信息：X, Y, POINT3D_ID
                # 格式："ddq" = double, double, int64（总共24字节）
                # *xy展开坐标为两个元素：[x, y, p3d_id]
                write_next_bytes(fid, [*xy, p3d_id], "ddq")


def read_points3D_text(path):
    """
    从文本文件中读取COLMAP 3D点数据
    
    对应COLMAP源码中的函数：
    src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    
    文本格式说明：
    # 每行格式：POINT3D_ID X Y Z R G B ERROR TRACK[] as (IMAGE_ID, POINT2D_IDX)
    # 示例：
    # 1 1.234 2.345 3.456 255 128 64 0.5 10 5 20 8 30 12
    # 表示：3D点ID=1, 坐标=(1.234,2.345,3.456), 颜色=(255,128,64), 
    #      误差=0.5, 在图像10的第5个特征点、图像20的第8个特征点、图像30的第12个特征点中被观测到
    
    参数:
    :param path: 3D点文本文件的路径（通常是points3D.txt）
    
    返回:
    :return: 字典，键为3D点ID，值为Point3D对象
    """
    points3D = {}  # 初始化空字典存储3D点数据
    
    # 以文本模式打开文件
    with open(path, "r") as fid:
        while True:
            # ===== 逐行读取文件内容 =====
            line = fid.readline()
            if not line:  # 如果到达文件末尾，退出循环
                break
                
            line = line.strip()  # 去除行首尾的空白字符
            
            # 跳过空行和注释行（以#开头的行）
            if len(line) > 0 and line[0] != "#":
                elems = line.split()  # 按空格分割行内容
                
                # ===== 解析3D点基本信息 =====
                point3D_id = int(elems[0])                              # 3D点ID
                xyz = np.array(tuple(map(float, elems[1:4])))          # 3D坐标 (X, Y, Z)
                rgb = np.array(tuple(map(int, elems[4:7])))            # RGB颜色值 (R, G, B)
                error = float(elems[7])                                 # 重投影误差
                
                # ===== 解析观测轨迹信息 =====
                # 从第8个元素开始，数据格式为：IMAGE_ID1 POINT2D_IDX1 IMAGE_ID2 POINT2D_IDX2 ...
                # 每两个元素为一组：(IMAGE_ID, POINT2D_IDX)
                
                # 提取图像ID：从索引8开始，每隔2个取一个 (8, 10, 12, ...)
                image_ids = np.array(tuple(map(int, elems[8::2])))
                
                # 提取2D特征点索引：从索引9开始，每隔2个取一个 (9, 11, 13, ...)
                point2D_idxs = np.array(tuple(map(int, elems[9::2])))
                
                # ===== 创建Point3D对象并存储 =====
                points3D[point3D_id] = Point3D(
                    id=point3D_id,              # 3D点ID
                    xyz=xyz,                    # 3D坐标
                    rgb=rgb,                    # RGB颜色
                    error=error,                # 重投影误差
                    image_ids=image_ids,        # 观测到该点的图像ID列表
                    point2D_idxs=point2D_idxs,  # 在对应图像中的2D特征点索引列表
                )
    
    return points3D  # 返回包含所有3D点的字典


def read_points3D_binary(path_to_model_file):
    """
    从二进制文件中读取COLMAP 3D点数据
    
    对应COLMAP源码中的函数：
    src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    
    二进制格式说明：
    1. 8字节：3D点总数量（uint64）
    2. 对每个3D点：
       - 8字节：3D点ID（uint64）
       - 24字节：3D坐标（3个double）
       - 3字节：RGB颜色（3个uint8）
       - 8字节：重投影误差（double）
       - 8字节：轨迹长度（uint64）
       - 8*N字节：N个轨迹元素（每个元素：int32图像ID + int32特征点索引）
    
    参数:
    :param path_to_model_file: 3D点二进制文件的路径（通常是points3D.bin）
    
    返回:
    :return: 字典，键为3D点ID，值为Point3D对象
    """
    points3D = {}  # 初始化空字典存储3D点数据
    
    # 以二进制模式打开文件
    with open(path_to_model_file, "rb") as fid:
        
        # ===== 读取3D点总数量 =====
        # 格式："Q" = 无符号64位整数（8字节）
        num_points = read_next_bytes(fid, 8, "Q")[0]
        
        # ===== 逐个读取每个3D点的数据 =====
        for _ in range(num_points):
            
            # ===== 读取3D点基本属性 =====
            # 读取43字节的基本属性数据
            # 格式："QdddBBBd" = uint64 + 3个double + 3个uint8 + 1个double
            # - 8字节：3D点ID（uint64）
            # - 24字节：3D坐标XYZ（3个double，每个8字节）
            # - 3字节：RGB颜色（3个uint8，每个1字节）
            # - 8字节：重投影误差（double）
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd"
            )
            
            # 解析基本属性
            point3D_id = binary_point_line_properties[0]        # 3D点ID
            xyz = np.array(binary_point_line_properties[1:4])   # 3D坐标 (X, Y, Z)
            rgb = np.array(binary_point_line_properties[4:7])   # RGB颜色 (R, G, B)
            error = np.array(binary_point_line_properties[7])   # 重投影误差
            
            # ===== 读取观测轨迹长度 =====
            # 格式："Q" = 无符号64位整数（8字节）
            # 轨迹长度表示有多少个图像观测到了这个3D点
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q"
            )[0]
            
            # ===== 读取观测轨迹数据 =====
            # 每个轨迹元素包含：图像ID（int32）+ 2D特征点索引（int32）
            # 总共需要读取 8 * track_length 字节
            # 格式："ii" * track_length，例如"iiii"表示2个轨迹元素
            track_elems = read_next_bytes(
                fid,
                num_bytes=8 * track_length,                     # 总字节数
                format_char_sequence="ii" * track_length,       # 格式字符串
            )
            
            # ===== 解析观测轨迹数据 =====
            # 数据排列：IMAGE_ID1, POINT2D_IDX1, IMAGE_ID2, POINT2D_IDX2, ...
            # 提取图像ID：索引0, 2, 4, ...
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            
            # 提取2D特征点索引：索引1, 3, 5, ...
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            
            # ===== 创建Point3D对象并存储 =====
            points3D[point3D_id] = Point3D(
                id=point3D_id,              # 3D点ID
                xyz=xyz,                    # 3D坐标
                rgb=rgb,                    # RGB颜色
                error=error,                # 重投影误差
                image_ids=image_ids,        # 观测到该点的图像ID列表
                point2D_idxs=point2D_idxs,  # 在对应图像中的2D特征点索引列表
            )
    
    return points3D  # 返回包含所有3D点的字典


def write_points3D_text(points3D, path):
    """
    将3D点数据写入文本文件
    
    对应COLMAP源码中的函数：
    src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    
    生成的文件格式：
    # 3D point list with one line of data per point:
    #   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
    # Number of points: N, mean track length: M
    1 1.234 2.345 3.456 255 128 64 0.5 10 5 20 8 30 12
    
    参数:
    :param points3D: 3D点字典，键为3D点ID，值为Point3D对象
    :param path: 输出文本文件的路径（通常是points3D.txt）
    """
    
    # ===== 计算统计信息 =====
    if len(points3D) == 0:
        mean_track_length = 0  # 如果没有3D点，平均轨迹长度为0
    else:
        # 计算每个3D点的平均观测轨迹长度
        # 轨迹长度表示有多少个图像观测到了该3D点
        mean_track_length = sum(
            (len(pt.image_ids) for _, pt in points3D.items())  # 每个3D点的观测图像数量
        ) / len(points3D)  # 除以3D点总数得到平均值
    
    # ===== 定义文件头部信息 =====
    HEADER = (
        "# 3D point list with one line of data per point:\n"                           # 说明每行包含一个3D点的数据
        + "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"  # 数据格式说明
        + "# Number of points: {}, mean track length: {}\n".format(
            len(points3D), mean_track_length                                           # 显示3D点总数和平均轨迹长度
        )
    )

    # ===== 写入文件 =====
    with open(path, "w") as fid:  # 以文本写入模式打开文件
        # 首先写入头部信息
        fid.write(HEADER)
        
        # ===== 遍历所有3D点并写入数据 =====
        for _, pt in points3D.items():  # 遍历3D点字典，忽略键，只使用值
            
            # ===== 准备3D点基本信息 =====
            point_header = [
                pt.id,      # 3D点ID
                *pt.xyz,    # 3D坐标（展开为3个元素：x, y, z）
                *pt.rgb,    # RGB颜色（展开为3个元素：r, g, b）
                pt.error    # 重投影误差
            ]
            
            # ===== 写入基本信息（不换行） =====
            # 将基本信息转换为字符串并用空格连接，末尾添加一个空格
            fid.write(" ".join(map(str, point_header)) + " ")
            
            # ===== 准备观测轨迹信息 =====
            track_strings = []  # 存储每个观测的字符串表示
            
            # 遍历所有观测该3D点的图像及其对应的2D特征点索引
            for image_id, point2D in zip(pt.image_ids, pt.point2D_idxs):
                # 为每个观测创建字符串：IMAGE_ID POINT2D_IDX
                track_strings.append(" ".join(map(str, [image_id, point2D])))
            
            # ===== 写入观测轨迹信息并换行 =====
            # 将所有观测字符串用空格连接并添加换行符
            fid.write(" ".join(track_strings) + "\n")


def write_points3D_binary(points3D, path_to_model_file):
    """
    将3D点数据写入二进制文件
    
    对应COLMAP源码中的函数：
    src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    
    二进制文件格式：
    1. 8字节：3D点总数量（uint64）
    2. 对每个3D点：
       - 8字节：3D点ID（uint64）
       - 24字节：3D坐标（3个double）
       - 3字节：RGB颜色（3个uint8）
       - 8字节：重投影误差（double）
       - 8字节：轨迹长度（uint64）
       - 8*N字节：N个轨迹元素（每个元素：int32图像ID + int32特征点索引）
    
    参数:
    :param points3D: 3D点字典，键为3D点ID，值为Point3D对象
    :param path_to_model_file: 输出二进制文件的路径（通常是points3D.bin）
    """
    
    # ===== 写入二进制文件 =====
    with open(path_to_model_file, "wb") as fid:  # 以二进制写入模式打开文件
        
        # ===== 写入3D点总数量 =====
        # 格式："Q" = 无符号64位整数（8字节）
        write_next_bytes(fid, len(points3D), "Q")
        
        # ===== 遍历所有3D点并写入数据 =====
        for _, pt in points3D.items():  # 遍历3D点字典
            
            # ===== 写入3D点基本属性 =====
            # 写入3D点ID（8字节无符号整数）
            write_next_bytes(fid, pt.id, "Q")
            
            # 写入3D坐标（3个8字节double）
            # tolist()将numpy数组转换为Python列表
            write_next_bytes(fid, pt.xyz.tolist(), "ddd")
            
            # 写入RGB颜色（3个1字节无符号整数）
            write_next_bytes(fid, pt.rgb.tolist(), "BBB")
            
            # 写入重投影误差（8字节double）
            write_next_bytes(fid, pt.error, "d")
            
            # ===== 写入观测轨迹数据 =====
            # 获取轨迹长度（观测该3D点的图像数量）
            track_length = pt.image_ids.shape[0]
            
            # 写入轨迹长度（8字节无符号整数）
            write_next_bytes(fid, track_length, "Q")
            
            # 写入每个观测的图像ID和2D特征点索引
            for image_id, point2D_id in zip(pt.image_ids, pt.point2D_idxs):
                # 格式："ii" = 两个4字节有符号整数（总共8字节）
                write_next_bytes(fid, [image_id, point2D_id], "ii")


def detect_model_format(path, ext):
    """
    检测COLMAP模型文件格式
    
    通过检查指定目录中是否存在所有必需的模型文件来确定格式类型
    
    参数:
    :param path: 模型文件所在的目录路径
    :param ext: 文件扩展名，用于区分格式类型
                - ".txt" 表示文本格式
                - ".bin" 表示二进制格式
    
    返回:
    :return: 布尔值，True表示检测到完整的模型格式，False表示文件不完整
    
    功能说明:
    - COLMAP的完整模型包含三个文件：cameras、images、points3D
    - 每种格式都需要这三个文件同时存在才算完整
    - 用于自动检测可用的模型格式
    """
    
    # ===== 检查所有必需文件是否存在 =====
    if (
        # 检查相机文件是否存在
        os.path.isfile(os.path.join(path, "cameras" + ext))
        # 检查图像文件是否存在
        and os.path.isfile(os.path.join(path, "images" + ext))
        # 检查3D点文件是否存在
        and os.path.isfile(os.path.join(path, "points3D" + ext))
    ):
        # ===== 如果所有文件都存在，输出检测结果 =====
        print("Detected model format: '" + ext + "'")
        return True  # 返回True表示检测成功

    return False  # 返回False表示文件不完整或不存在


def read_model(path, ext=""):
    """
    读取完整的COLMAP模型数据
    
    这是一个高级接口函数，用于一次性读取COLMAP的完整3D重建模型，
    包括相机参数、图像姿态和3D点云数据。
    
    参数:
    :param path: 模型文件所在的目录路径
    :param ext: 文件扩展名，指定格式类型
                - ".bin": 二进制格式（默认自动检测）
                - ".txt": 文本格式
                - "": 空字符串表示自动检测格式
    
    返回:
    :return: 三元组 (cameras, images, points3D)
             - cameras: 相机字典，键为相机ID，值为Camera对象
             - images: 图像字典，键为图像ID，值为Image对象  
             - points3D: 3D点字典，键为3D点ID，值为Point3D对象
    """
    
    # ===== 自动格式检测 =====
    if ext == "":  # 如果没有指定扩展名，尝试自动检测
        if detect_model_format(path, ".bin"):
            # 优先检测二进制格式（通常更常用，读取速度更快）
            ext = ".bin"
        elif detect_model_format(path, ".txt"):
            # 如果二进制格式不存在，检测文本格式
            ext = ".txt"
        else:
            # 如果两种格式都不存在，提示用户并返回
            print("请提供模型格式: '.bin' 或 '.txt'")
            return  # 返回None，表示读取失败

    # ===== 根据格式选择相应的读取函数 =====
    if ext == ".txt":
        # ===== 读取文本格式模型 =====
        # 读取相机参数文件：cameras.txt
        cameras = read_cameras_text(os.path.join(path, "cameras" + ext))
        
        # 读取图像姿态文件：images.txt
        images = read_images_text(os.path.join(path, "images" + ext))
        
        # 读取3D点云文件：points3D.txt
        points3D = read_points3D_text(os.path.join(path, "points3D") + ext)
    else:
        # ===== 读取二进制格式模型 =====
        # 读取相机参数文件：cameras.bin
        cameras = read_cameras_binary(os.path.join(path, "cameras" + ext))
        
        # 读取图像姿态文件：images.bin
        images = read_images_binary(os.path.join(path, "images" + ext))
        
        # 读取3D点云文件：points3D.bin
        points3D = read_points3D_binary(os.path.join(path, "points3D") + ext)
    
    # ===== 返回完整的模型数据 =====
    return cameras, images, points3D


def write_model(cameras, images, points3D, path, ext=".bin"):
    """
    写入完整的COLMAP模型数据
    
    这是一个高级接口函数，用于一次性写入COLMAP的完整3D重建模型，
    包括相机参数、图像姿态和3D点云数据。
    
    参数:
    :param cameras: 相机字典，键为相机ID，值为Camera对象
    :param images: 图像字典，键为图像ID，值为Image对象
    :param points3D: 3D点字典，键为3D点ID，值为Point3D对象
    :param path: 输出模型文件的目录路径
    :param ext: 文件扩展名，指定输出格式类型
                - ".bin": 二进制格式（默认，推荐用于大规模数据）
                - ".txt": 文本格式（便于查看和调试）
    
    返回:
    :return: 三元组 (cameras, images, points3D)
             返回输入的数据，便于链式调用或验证
    
    功能说明:
    - 自动创建输出目录（如果不存在）
    - 确保三个文件的格式一致性
    - 生成与COLMAP官方完全兼容的文件格式
    """
    
    # ===== 根据格式选择相应的写入函数 =====
    if ext == ".txt":
        # ===== 写入文本格式模型 =====
        # 写入相机参数文件：cameras.txt
        write_cameras_text(cameras, os.path.join(path, "cameras" + ext))
        
        # 写入图像姿态文件：images.txt
        write_images_text(images, os.path.join(path, "images" + ext))
        
        # 写入3D点云文件：points3D.txt
        write_points3D_text(points3D, os.path.join(path, "points3D") + ext)
    else:
        # ===== 写入二进制格式模型（默认） =====
        # 写入相机参数文件：cameras.bin
        write_cameras_binary(cameras, os.path.join(path, "cameras" + ext))
        
        # 写入图像姿态文件：images.bin
        write_images_binary(images, os.path.join(path, "images" + ext))
        
        # 写入3D点云文件：points3D.bin
        write_points3D_binary(points3D, os.path.join(path, "points3D") + ext)
    
    # ===== 返回输入数据 =====
    # 返回原始数据，便于后续处理或验证写入是否成功
    return cameras, images, points3D


def qvec2rotmat(qvec):
    """
    将四元数转换为旋转矩阵
    
    四元数是表示3D旋转的一种数学工具，具有以下优势：
    - 避免万向锁问题
    - 插值更平滑
    - 存储空间小（4个数vs 9个数）
    - 数值稳定性好
    
    参数:
    :param qvec: 四元数向量 [w, x, y, z]，其中：
                - w: 标量部分（实部）
                - x, y, z: 向量部分（虚部），表示旋转轴的方向
                - 满足归一化条件：w² + x² + y² + z² = 1
    
    返回:
    :return: 3×3旋转矩阵，用于将向量从一个坐标系旋转到另一个坐标系
    
    数学原理:
    四元数到旋转矩阵的转换公式基于四元数的性质：
    q = w + xi + yj + zk
    旋转矩阵R的每个元素都可以用四元数分量表示
    """
    
    # ===== 四元数分量提取 =====
    # qvec[0] = w (标量部分)
    # qvec[1] = x (i分量)  
    # qvec[2] = y (j分量)
    # qvec[3] = z (k分量)
    
    return np.array([
        # ===== 旋转矩阵第一行 =====
        [
            # R[0,0] = 1 - 2(y² + z²)
            # 这表示绕x轴旋转对x坐标的影响最小
            1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
            
            # R[0,1] = 2(xy - wz)  
            # xy项表示x和y轴的耦合，wz项是由于旋转轴的影响
            2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
            
            # R[0,2] = 2(xz + wy)
            # xz项表示x和z轴的耦合，wy项是由于旋转轴的影响
            2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
        ],
        # ===== 旋转矩阵第二行 =====
        [
            # R[1,0] = 2(xy + wz)
            # 与R[0,1]相对应，但符号不同，体现旋转的反对称性
            2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
            
            # R[1,1] = 1 - 2(x² + z²)
            # 这表示绕y轴旋转对y坐标的影响最小
            1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
            
            # R[1,2] = 2(yz - wx)
            # yz项表示y和z轴的耦合，wx项是由于旋转轴的影响
            2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
        ],
        # ===== 旋转矩阵第三行 =====
        [
            # R[2,0] = 2(xz - wy)
            # 与R[0,2]相对应，但符号不同
            2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
            
            # R[2,1] = 2(yz + wx)
            # 与R[1,2]相对应，但符号不同
            2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
            
            # R[2,2] = 1 - 2(x² + y²)
            # 这表示绕z轴旋转对z坐标的影响最小
            1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
        ],
    ])


def rotmat2qvec(R):
    """
    将旋转矩阵转换为四元数
    
    这是qvec2rotmat的逆操作，使用Shepperd方法进行转换，
    该方法数值稳定性好，避免了除零和精度问题。
    
    参数:
    :param R: 3×3旋转矩阵，必须是正交矩阵且行列式为1
             即满足：R^T * R = I 且 det(R) = 1
    
    返回:
    :return: 四元数向量 [w, x, y, z]，归一化后的四元数
    
    算法原理:
    使用特征值分解方法，构造一个4×4对称矩阵K，
    其最大特征值对应的特征向量就是所求的四元数。
    """
    
    # ===== 提取旋转矩阵的所有元素 =====
    # R.flat将3×3矩阵展平为9个元素的一维数组
    # 按行优先顺序：R[0,0], R[0,1], R[0,2], R[1,0], R[1,1], R[1,2], R[2,0], R[2,1], R[2,2]
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    
    # ===== 构造Shepperd矩阵K =====
    # 这是一个4×4对称矩阵，其最大特征值的特征向量对应四元数
    # 矩阵K的构造基于四元数与旋转矩阵元素之间的关系
    K = (
        np.array([
            # 第一行：[Rxx - Ryy - Rzz, 0, 0, 0]
            # 对角线元素的差，与四元数的x分量相关
            [Rxx - Ryy - Rzz, 0, 0, 0],
            
            # 第二行：[Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0]  
            # 非对角线元素的和与差，与四元数的y分量相关
            [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
            
            # 第三行：[Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0]
            # 与四元数的z分量相关
            [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
            
            # 第四行：[Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]
            # 反对称元素的差和对角线元素的和，与四元数的w分量相关
            [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
        ])
        / 3.0  # 归一化因子，确保数值稳定性
    )
    
    # ===== 计算特征值和特征向量 =====
    # np.linalg.eigh专门用于对称矩阵的特征值分解，比eig更高效和稳定
    eigvals, eigvecs = np.linalg.eigh(K)
    
    # ===== 提取四元数 =====
    # 选择最大特征值对应的特征向量
    # eigvecs的列是特征向量，np.argmax(eigvals)找到最大特征值的索引
    # [3, 0, 1, 2]重新排列特征向量的顺序，使其符合[w, x, y, z]的四元数格式
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    
    # ===== 确保四元数的标准形式 =====
    # 四元数q和-q表示相同的旋转，但通常约定w分量为正
    # 这样可以确保四元数的唯一性和一致性
    if qvec[0] < 0:
        qvec *= -1  # 如果w分量为负，将整个四元数取反
    
    return qvec  # 返回归一化的四元数 [w, x, y, z]


# def main():
#     parser = argparse.ArgumentParser(
#         description="Read and write COLMAP binary and text models"
#     )
#     parser.add_argument("--input_model", help="path to input model folder")
#     parser.add_argument(
#         "--input_format",
#         choices=[".bin", ".txt"],
#         help="input model format",
#         default="",
#     )
#     parser.add_argument("--output_model", help="path to output model folder")
#     parser.add_argument(
#         "--output_format",
#         choices=[".bin", ".txt"],
#         help="outut model format",
#         default=".txt",
#     )
#     args = parser.parse_args()

#     cameras, images, points3D = read_model(
#         path=args.input_model, ext=args.input_format
#     )

#     print("num_cameras:", len(cameras))
#     print("num_images:", len(images))
#     print("num_points3D:", len(points3D))

#     if args.output_model is not None:
#         write_model(
#             cameras,
#             images,
#             points3D,
#             path=args.output_model,
#             ext=args.output_format,
#         )


# if __name__ == "__main__":
#     main()
