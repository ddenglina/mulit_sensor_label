import pandas as pd
import math
import open3d as o3d
import numpy as np


def parse_3d_annotations(txt_path):
    """解析3D标签TXT文件，返回与JSON解析函数结构一致的列表"""
    # 读取TXT文件（使用pandas处理表头和数据）
    df = pd.read_csv(txt_path)
    
    cubes = []
    for i, row in df.iterrows():  # i为索引，作为id
        # 旋转角度从度转换为弧度（与JSON中rotation的弧度单位保持一致）
        rotation_rad = [
            math.radians(row['R_x']),
            math.radians(row['R_y']),
            math.radians(row['R_z'])
        ]
        
        cube = {
            'label': row['label'],  # 类别标签
            'center': [row['center_x'], row['center_y'], row['center_z']],  # 中心点坐标
            'dimensions': [row['extent_x'], row['extent_y'], row['extent_z']],  # 尺寸（对应extent）
            "rotation": rotation_rad  # 旋转角度（弧度）
        }
        cubes.append(cube)
    
    return cubes


def create_bounding_box(center, dimensions, rotation):
    """
    创建3D边界框
    
    参数:
        center: 边界框中心点坐标 [x, y, z]
        dimensions: 边界框尺寸 [长度, 宽度, 高度]
        rotation: 边界框旋转角度 [x, y, z] (弧度)
    
    返回:
        open3d.geometry.OrientedBoundingBox对象
    """
    # 创建轴对齐边界框
    bbox = o3d.geometry.OrientedBoundingBox(
        center=np.array(center),
        extent=np.array(dimensions),
        # R = euler2rot(rotation),
        R = euler2rot(rotation, order='xyz')
    )
    
    return bbox

def euler2rot(euler, order='xyz'):
    """
    欧拉角转旋转矩阵
    :param euler: 欧拉角 (α, β, γ)，单位：弧度
    :param order: 旋转顺序（如 'xyz', 'zyx', 'xzy' 等）
    :return: 3x3 旋转矩阵
    """
    alpha, beta, gamma = euler
    # 定义单个轴的旋转矩阵
    def Rx(θ):
        return np.array([[1, 0, 0],
                         [0, np.cos(θ), -np.sin(θ)],
                         [0, np.sin(θ), np.cos(θ)]])
    def Ry(θ):
        return np.array([[np.cos(θ), 0, np.sin(θ)],
                         [0, 1, 0],
                         [-np.sin(θ), 0, np.cos(θ)]])
    def Rz(θ):
        return np.array([[np.cos(θ), -np.sin(θ), 0],
                         [np.sin(θ), np.cos(θ), 0],
                         [0, 0, 1]])
    
    # 根据旋转顺序计算总旋转矩阵（矩阵乘法顺序与旋转顺序一致）
    rot_map = {'x': Rx, 'y': Ry, 'z': Rz}
    R = np.eye(3)  # 初始化为单位矩阵
    for axis, angle in zip(order, [alpha, beta, gamma]):
        R = rot_map[axis](angle) @ R  # 矩阵乘法（@ 等价于 np.matmul）
    return R


pcd_file = '/mnt/dln/data/datasets/0915/make_label_raw/lab_430/pcd/1.pcd'
frame_pcd = o3d.io.read_point_cloud(pcd_file)

txt_path = '/mnt/dln/data/datasets/0915/make_label_raw/lab_430/3D_label/1.txt'
cubes = parse_3d_annotations(txt_path)


vis_objects = []
for id,cube in enumerate(cubes):
    # 提取原始边界框参数
    center = cube["center"]
    dimensions = cube["dimensions"]
    original_rotation = cube["rotation"]
    # original_rotation = [0,0,0]
    # print(original_rotation)
    label_name = cube["label"]

    bbox = create_bounding_box(center, dimensions, original_rotation)
    vis_objects.append(bbox)

o3d.visualization.draw_geometries([frame_pcd]+vis_objects)
