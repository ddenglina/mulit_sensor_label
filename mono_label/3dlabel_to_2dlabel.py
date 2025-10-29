import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import os
from PIL import Image
from matplotlib.patches import Rectangle
import cv2
import pandas as pd
import math
import open3d as o3d


# --------------------------
# 1. 配置：类别颜色映射和名称映射（核心新增）
# --------------------------
# 类别ID -> BGR颜色（确保颜色区分明显，OpenCV用BGR格式）
class_color_map = {
    "car": (0, 0, 255),    # 类别0：红色（如car）
    "chair": (0, 255, 0),    # 类别1：绿色（如pedestrian）
    "table": (255, 0, 0),    # 类别2：蓝色（如cyclist）
    "robot": (0, 255, 255),  # 类别3：黄色（如truck）
    "robot_arm": (255, 0, 128),  # 类别4：紫色（如bus）
    "trash_can": (255, 0, 255), # 类别4：紫色（如bus）
    "screen": (255, 255, 0), # 类别5：黄色（如screen）
    "cabinet":(128, 128, 0),    # 类别6：\(如cabinet）
    "sitting_people": (0, 128, 0),    # 类别7：绿色（如sitting_people）
    "potted_plant": (128, 0, 0),    # 类别8：棕色（如potted_plant）
    "computer": (0, 0, 128),    # 类别9：蓝色（如computer）
    "storage_rack": (128, 128, 128),    # 类别10：灰色（如storage_rack）
    "plastic_stool": (255, 128, 0),    # 类别11：橙色（如plastic_stool）
    "fan": (50, 128, 0),
    "white_board": (50, 50, 50),    # 类别12：白色（如white_board）
    # 可根据实际类别扩展...
}

# 类别ID -> BGR颜色（确保颜色区分明显，OpenCV用BGR格式）
min_points_list = {
    "car": 130,    # 类别0：红色（如car）
    "chair": 35,    # 类别1：绿色（如pedestrian）
    "table": 50,    # 类别2：蓝色（如cyclist）
    "robot": 35,  # 类别3：黄色（如truck）
    "robot_arm": 50,  # 类别3：黄色（如truck）
    "trash_can": 35, # 类别4：紫色（如bus）
    "screen": 35, # 类别5：黄色（如screen）
    "cabinet": 35,    # 类别6：\(如cabinet）
    "sitting_people": 50,    # 类别7：绿色（如sitting_people）
    "potted_plant": 35,    # 类别8：棕色（如potted_plant）
    "computer": 35,    # 类别9：蓝色（如computer）
    "storage_rack": 50,    # 类别10：灰色（如storage_rack）
    "plastic_stool": 35,    # 类别11：橙色（如plastic_stool）
    "fan": 50,
    "white_board": 35,    # 类别12：白色（如white_board）
    # 可根据实际类别扩展...
}

# 有效面积占整张图像比，小物體小佔比
valid_ratio_list = {
    "car": 0.4,    # 类别0：红色（如car）
    "chair": 0.3,    # 类别1：绿色（如pedestrian）
    "table": 0.5,    # 类别2：蓝色（如cyclist）
    "robot": 0.1,  # 类别3：黄色（如truck）
    "robot_arm": 0.4,  # 类别4：紫色（如bus）
    "trash_can": 0.1, # 类别4：紫色（如bus）
    "screen": 0.1, # 类别5：黄色（如screen）
    "cabinet": 0.1,    # 类别6：\(如cabinet）
    "sitting_people": 0.2,    # 类别7：绿色（如sitting_people）
    "potted_plant": 0.1,    # 类别8：棕色（如potted_plant）
    "computer": 0.1,    # 类别9：蓝色（如computer）
    "storage_rack": 0.2,    # 类别10：灰色（如storage_rack）
    "plastic_stool":0.1,
    "fan": 0.1,
    "white_board": 0.3,    # 类别12：白色（如white_board）
}

# 截斷有效面積占比,小物體大佔比
trunc_area_ratio_list = {
    "car": 0.2,    # 类别0：红色（如car）
    "chair": 0.5,    # 类别1：绿色（如pedestrian）
    "table": 0.3,    # 类别2：蓝色（如cyclist）
    "robot": 0.4,  # 类别3：黄色（如truck）
    "robot_arm": 0.2,  # 类别4：紫色（如bus）
    "trash_can": 0.4, # 类别4：紫色（如bus）
    "screen": 0.5, # 类别5：黄色（如screen）
    "cabinet": 0.5,    # 类别6：\(如cabinet）
    "sitting_people": 0.4,    # 类别7：绿色（如sitting_people）
    "potted_plant": 0.4,    # 类别8：棕色（如potted_plant）
    "computer": 0.4,    # 类别9：蓝色（如computer）
    "storage_rack": 0.4,    # 类别10：灰色（如storage_rack）
    "plastic_stool":0.4,
    "fan": 0.4,
    "white_board": 0.5,    # 类别12：白色（如white_board）
}

def get_common_view_points(points, K, R_lidar2cam, T_lidar2cam, img_width, img_height):
    """提取共视区点云及投影坐标"""
    # 雷达点→相机坐标系
    points_hom = np.hstack([points, np.ones((points.shape[0], 1))])  # (N,4)
    extrinsic = np.hstack([R_lidar2cam, T_lidar2cam.reshape(3,1)])    # (3,4)
    points_cam = (extrinsic @ points_hom.T).T                         # (N,3)：(X,Y,Z)
    
    # 筛选相机前方的点（Z>0）
    front_mask = points_cam[:, 2] > 1e-6
    points_cam_valid = points_cam[front_mask]
    points_valid = points[front_mask]
    
    # 投影到图像平面
    Z = points_cam_valid[:, 2].reshape(-1, 1)
    X = points_cam_valid[:, 0].reshape(-1, 1)
    Y = points_cam_valid[:, 1].reshape(-1, 1)
    u = (K[0,0] * X / Z + K[0,2]).flatten()  # 像素u
    v = (K[1,1] * Y / Z + K[1,2]).flatten()  # 像素v
    
    # 筛选图像范围内的点
    img_mask = (u >= 0) & (u < img_width) & (v >= 0) & (v < img_height)
    common_points = points_valid[img_mask]
    uvs = np.column_stack([u[img_mask], v[img_mask]]).astype(int)
    return common_points, uvs


# 1. 解析3D点云标注的JSON文件
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


# 3. 定义相机内参 (手动设置)
def get_camera_intrinsics():
    """手动设置相机内参"""

    # geoscan d435i
    cam_fx = 605.231000
    cam_fy = 605.133900  # 焦距y
    cam_cx = 320.931700   # 主点x坐标
    cam_cy = 253.091400   # 主点y坐标
    width = 640 # 图像宽度
    height = 480 # 图像高度

    camera_to_lidar = np.array([
        [-0.028191, -0.9996, 0.000532, 0.033057],
        [0.430632, -0.012625, -0.902439, -0.116977],
        [0.902087, -0.025211, 0.430817, 0.013525],
        [0, 0, 0, 1]
        ], dtype=np.float64)



    # width = 512
    # height = 384
    # #0
    # cam_fx = 491.61243079/2
    # cam_fy = 491.59793421/2
    # cam_cx = 509.19576525/2
    # cam_cy = 390.55245804/2
    # camera_to_lidar = np.array([
    #     [-0.720207,  -0.693753,   0.002868,  0.090756],
    #     [-0.012521,   0.008864,  -0.999882, -0.006529],
    #     [0.693646,  -0.720158,  -0.015070, -0.084491],
    #     [0, 0, 0, 1]
    #     ], dtype=np.float64)

    # 1
    # fx=491.30059484/2
    # fy=491.52032513/2
    # cx=535.07657118/2
    # cy=404.55184217/2
    # camera_to_lidar = np.array([
    #     [0.707555,  -0.706598,   0.009238,  0.054002],
    #     [-0.004619,  -0.017697,  -0.999833, -0.007218],
    #     [0.706643,   0.707394,  -0.015785,  -0.088251],
    #     [0, 0, 0, 1]
    #     ], dtype=np.float64)

    #2
    # cam_fx = 504.71027203/2  
    # cam_fy = 505.94199486/2
    # cam_cx = 558.67274185/2
    # cam_cy = 399.34953343/2

    # camera_to_lidar = np.array([
    #      [ -0.693995,   0.719643,   0.022003,-0.005720],
    #     [  -0.032811,  -0.001084,  -0.999461,-0.078270],
    #     [  -0.719232,  -0.694343,   0.024365,-0.030108],
    #     [ 0, 0, 0, 1]], dtype=np.float64) 

    # 3
    # cam_fx = 492.21619642/2
    # cam_fy = 491.75080876/2
    # cam_cx = 531.39937227/2
    # cam_cy = 394.35920146/2
    # camera_to_lidar = np.array([
    #     [  0.726027,   0.687592,  -0.010122, 0.079652],
    #     [ 0.001577,  -0.016384,  -0.999865, 0.014322],
    #     [-0.687665,   0.725912,  -0.012979,-0.068272],
    #     [ 0, 0, 0, 1]], dtype=np.float64) 
    
    # 内参矩阵
    K = np.array([
        [cam_fx, 0, cam_cx],
        [0, cam_fy, cam_cy],
        [0, 0, 1]
    ])

    return K, width, height, camera_to_lidar

# 4. 坐标转换和投影函数
def quaternion_to_rotation_matrix(q):
    """将四元数转换为旋转矩阵"""
    w, x, y, z = q
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])

def euler_to_rotation_matrix(euler):
    """将欧拉角转换为旋转矩阵 (XYZ顺序)"""
    rx, ry, rz = euler
    
    # 绕X轴旋转
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    
    # 绕Y轴旋转
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    
    # 绕Z轴旋转
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])
    
    # 组合旋转矩阵
    R = Rz @ Ry @ Rx
    return R

def compute_bounding_box(points_2d):
    """计算2D点集的最小外接矩形"""
    min_x = np.min(points_2d[:, 0])
    max_x = np.max(points_2d[:, 0])
    min_y = np.min(points_2d[:, 1])
    max_y = np.max(points_2d[:, 1])
    
    # 计算中心点、宽度和高度
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    width = max_x - min_x
    height = max_y - min_y
    
    return {
        'min_x': min_x, 'max_x': max_x,
        'min_y': min_y, 'max_y': max_y,
        'center_x': center_x, 'center_y': center_y,
        'width': width, 'height': height
    }


def get_cube_vertices(center, dimensions, rotation):
    """计算3D立方体的8个顶点坐标"""
    l, w, h = dimensions
    # 局部坐标系下的8个顶点
    vertices_local = np.array([
        [l/2, w/2, h/2],   # 前右上
        [l/2, w/2, -h/2],  # 后右上
        [l/2, -w/2, h/2],  # 前右下
        [l/2, -w/2, -h/2], # 后右下
        [-l/2, w/2, h/2],  # 前左上
        [-l/2, w/2, -h/2], # 后左上
        [-l/2, -w/2, h/2], # 前左下
        [-l/2, -w/2, -h/2] # 后左下
    ])
    
    # 应用旋转
    R = euler_to_rotation_matrix(rotation)
    vertices_rotated = np.dot(vertices_local, R.T)
    
    # 应用平移（移动到中心点）
    vertices_world = vertices_rotated + np.array(center)
    
    return vertices_world

def project_3d_to_2d(points_3d, K, R, t):
    """将3D点投影到2D图像平面"""
    points_3d_cam = R @ points_3d.T + t.reshape(3, 1)  # 3xN
    points_3d_cam = points_3d_cam.T  # 转置为Nx3
    
    # 将3D点云投影到2D图像平面
    points_2d_homogeneous = K @ points_3d_cam.T  # 3xN
    points_2d = points_2d_homogeneous[:2, :] / points_2d_homogeneous[2, :]  # 归一化
    points_2d = points_2d.T  # 转置为Nx2
    
    return points_2d


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
        R=euler2rot(rotation),
 
    )
    
    return bbox



# 5. 可视化函数
def visualize_projection_on_image(frame_pcd, cubes, K, cam2lid,image_path, save_2dlabel_path,vis_save_path,
                                  is_rotated_180=True, min_area_ratio=0.3, min_valid_abs_area=10):
    """在实际图像上可视化3D立方体的投影"""
    # 加载图像
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    # print(f"处理图像: {image_name} (尺寸: {width}x{height})")
    
    # 获取相机外参
    R = cam2lid[:3, :3]
    t = cam2lid[:3, 3]
    
    # 立方体的边连接关系
    edges = [
        [0, 1], [1, 3], [3, 2], [2, 0],  # 前面和后面
        [4, 5], [5, 7], [7, 6], [6, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]   # 连接前后
    ]
    
    # 存储当前图像的所有YOLO标签
    yolo_labels = []
    vis_objects = []


    # 处理每个立方体
    for cube in cubes:
        # print("cube:",cube)
        # 计算立方体顶点
        vertices = get_cube_vertices(
            cube['center'], 
            cube['dimensions'], 
            cube['rotation']
        )

        # 投影到2D图像
        vertices_2d = project_3d_to_2d(vertices, K, R, t)
        # 过滤掉在相机后方的点 (z < 0)
        points_cam = R @ vertices.T + np.array(t).reshape(3, 1)
        visible = points_cam[2, :] > 0
        if not np.any(visible):
            # print("continue")
            continue  # 完全不可见的立方体不处理
        
        # ## 增加框中是否有点云判断 #########################################################
        # 1.提取共视区点云
        filter_frame_pcd, _ = get_common_view_points(np.asarray(frame_pcd.points), K, R, t, width, height)
        # 2.判断共视区点云中是否有足够的点
        bbox = create_bounding_box(cube['center'], cube['dimensions'], cube['rotation'])        
        points_translated = np.asarray(filter_frame_pcd) - bbox.center
        points_local = np.dot(points_translated, bbox.R)  # 转换到局部坐标系
        half_extent = bbox.extent / 2
        in_bbox = np.all(np.abs(points_local) <= half_extent + 1e-6, axis=1)
        num_in_box = np.sum(in_bbox)
        
        # 如果点数 >= min_points，则保留该框
        if num_in_box < min_points_list[cube['label']]:
            continue  # 点数不足，跳过该框
        # ###########################################################


        vis_objects.append(bbox)  
        # 获取可见的2D顶点
        visible_vertices_2d = vertices_2d[visible]
        bbox = compute_bounding_box(visible_vertices_2d)

        # --------------------------
        # 标签处理
        min_x_original = bbox['min_x']
        min_y_original = bbox['min_y']
        clamped_width = bbox['width']  # 原始宽（未旋转图像中）
        clamped_height = bbox['height']  # 原始高（未旋转图像中）
        max_x_original = min_x_original + clamped_width
        max_y_original = min_y_original + clamped_height
        
        # 2. 计算原始框面积，防护面积为0的异常情况
        original_area = clamped_width * clamped_height
        # 2. 转换坐标到180度旋转后的图像
        if is_rotated_180:
            # 旋转后坐标（x和y轴均反转）
            min_x_rot = width - 1 - min_x_original
            max_x_rot = width - 1 - max_x_original
            min_y_rot = height - 1 - min_y_original
            max_y_rot = height - 1 - max_y_original
            
            # 旋转后min和max可能互换，需重新排序
            min_x_rot, max_x_rot = sorted([min_x_rot, max_x_rot])
            min_y_rot, max_y_rot = sorted([min_y_rot, max_y_rot])
        else:
            min_x_rot, max_x_rot = min_x_original, max_x_original
            min_y_rot, max_y_rot = min_y_original, max_y_original
        
        # --------------------------
        # 3. 核心：裁剪超出图像边界的坐标（新增步骤）
        # --------------------------
        # 裁剪x坐标到[0, width-1]
        min_x_clamped = np.clip(min_x_rot, 0, width - 1)
        max_x_clamped = np.clip(max_x_rot, 0, width - 1)
        min_x, max_x = (min_x_clamped, max_x_clamped) if min_x_clamped <= max_x_clamped else (max_x_clamped, min_x_clamped)
        # 裁剪y坐标到[0, height-1]
        min_y_clamped = np.clip(min_y_rot, 0, height - 1)
        max_y_clamped = np.clip(max_y_rot, 0, height - 1)
        min_y, max_y = (min_y_clamped, max_y_clamped) if min_y_clamped <= max_y_clamped else (max_y_clamped, min_y_clamped)
        
        # 计算裁剪后的有效宽高，过滤无效框
        # 5. 计算有效框面积及占比，过滤小占比框（核心优化）
        valid_width = max_x_clamped - min_x_clamped
        valid_height = max_y_clamped - min_y_clamped
        valid_area = valid_width * valid_height

        if original_area <= 0:
            # 原始框面积为0（无效），直接丢弃
            continue

        # 4. 计算面积占比
        area_ratio = valid_area / original_area
       

        max_area = height*width
        valid_ratio = valid_area / max_area # 有效面积占整张图像比
         # 过滤条件：有效面积为0 或 占比不足且 绝对面积太小
        if valid_area <= 0 or (area_ratio < trunc_area_ratio_list[cube['label']] and valid_ratio <= valid_ratio_list[cube['label']]):
            continue
        
        # 4. 绘制带颜色的矩形框（使用裁剪后的坐标）
        box_color = class_color_map.get(cube['label'], (128, 128, 128))  # 类别颜色
        class_name = cube['label']

        pt1 = (int(min_x_clamped), int(min_y_clamped))  # 裁剪后左上角
        pt2 = (int(max_x_clamped), int(max_y_clamped))  # 裁剪后右下角
        cv2.rectangle(img, pt1, pt2, box_color, 2)  # 仅绘制图像内的部分


        # 5. 绘制类别名称（适配裁剪后的位置）
        # 文本位置：裁剪后框的左上角上方（避免超出图像）
        if pt1[1] - 10 > 0:  # 上方有空间
            text_pos = (pt1[0], pt1[1] - 10)
        else:  # 上方无空间，放框内
            text_pos = (pt1[0], pt1[1] + 20)
        
        # 白色文本+黑色描边，确保清晰
        cv2.putText(img, class_name, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)  # 描边
        cv2.putText(img, class_name, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)  # 文本
        
        # 添加到YOLO标签列表
        yolo_line = f"{cube['label']} {pt1[0]} {pt1[1]} {pt2[0]} {pt2[1]}"
        yolo_labels.append(yolo_line)
            
    # print(f"标签文件路径: {save_2dlabel_path}")
    # 写入标签（若没有目标则创建空文件）
    with open(save_2dlabel_path, 'w', encoding='utf-8') as f:
        if yolo_labels:
            f.write('\n'.join(yolo_labels) + '\n')  # 每行一个目标
        else:
            f.write('')  # 无目标则为空文件

    # 图像和点云一起可视化
    image_name = os.path.basename(image_path)
    if image_name in ["02284.png","02926.png"]:
        cv2.imshow("2D Bounding Boxes", img)
        cv2.waitKey(0)
        o3d.visualization.draw_geometries([frame_pcd]+vis_objects)
        cv2.destroyAllWindows()
    
    if True:
        print(vis_save_path)
        cv2.imwrite(vis_save_path, img)
    return img

# 6. 主函数
def main(label_3d_dir, fixed_image_dir,label_2d_dir,vis_save_dir):

    for file_name in sorted(os.listdir(label_3d_dir)):
        txt_path = os.path.join(label_3d_dir, file_name)    
        num_str = file_name.split('.')[0]  # 提取数字部分
        image_name = f"{int(num_str):05d}"  # 转换为整数后格式化，结果："00111"
        image_name = image_name + ".png"
        img_path = os.path.join(fixed_image_dir, image_name)
        pcd_path = os.path.join(label_3d_dir.replace("3D_label","pcd"), file_name.replace(".txt",".pcd"))
        save_2dlabel_path = os.path.join(label_2d_dir, image_name[:-4]+".txt")
        vis_save_path = os.path.join(vis_save_dir, image_name)

        # 解析数据
        frame_pcd = o3d.io.read_point_cloud(pcd_path)
        cubes = parse_3d_annotations(txt_path)
        K, _, _,cam2lid = get_camera_intrinsics()
        # print(f"解析到 {len(cubes)} 个3D立方体")

        fig = visualize_projection_on_image(frame_pcd, cubes, K, cam2lid, img_path,save_2dlabel_path,vis_save_path,
                                             is_rotated_180=True)
        print(11111)

       

if __name__ == "__main__":

    # scene_dir = '/mnt/dln/data/datasets/0915/make_label_raw/lab_429/'  # 场景目录
    scene_dir = "/mnt/dln/data/datasets/0915/for_bev_test/"
    # scene_dir = '/mnt/dln/data/datasets/0915/for_bev_lidar_test/'  # 场景目录
    label_3d_dir = scene_dir + '/3D_label'
    fixed_image_dir = scene_dir + '/fixed_images'
    label_2d_dir = scene_dir + '/2D_label'
    vis_save_dir =  scene_dir + "fixed_images_with_label"
    if not os.path.exists(label_2d_dir):
        os.makedirs(label_2d_dir)
    if len(os.listdir(label_2d_dir)) > 0:
        print(f"警告：2D标签目录 {label_2d_dir} 已存在且非空，可能覆盖已有标签")
    
    if not os.path.exists(vis_save_dir):
        os.makedirs(vis_save_dir)

    main(label_3d_dir, fixed_image_dir,label_2d_dir,vis_save_dir) 

    