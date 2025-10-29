
# 环视图像2D标签构建
# 方案：基于纯lidar建图，利用环视-雷达外参标定、内参标定构建2D标签
# 输入：3D_label、images
# 输出：2D_label、可视化图像

import math
import os
import cv2
import numpy as np
import pandas as pd
import shutil
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



def get_camera_intrinsics_dict():
    """将相机参数构建成字典形式返回"""
    camera_dict = {}
    
    # 相机0参数
    camera_dict[0] = {
        'width': 512,
        'height': 384,
        'K': np.array([
            [491.61243079 / 2, 0, 509.19576525 / 2],
            [0, 491.59793421 / 2, 390.55245804 / 2],
            [0, 0, 1]
        ], dtype=np.float64),
        'camera_to_lidar': np.array([
            [-0.720207,  -0.693753,   0.002868,  0.090756],
            [-0.012521,   0.008864,  -0.999882, -0.006529],
            [0.693646,  -0.720158,  -0.015070, -0.084491],
            [0, 0, 0, 1]
        ], dtype=np.float64)
    }
    
    # 相机1参数
    camera_dict[1] = {
        'width': 512,
        'height': 384,
        'K': np.array([
            [491.30059484 / 2, 0, 535.07657118 / 2],
            [0, 491.52032513 / 2, 404.55184217 / 2],
            [0, 0, 1]
        ], dtype=np.float64),
        'camera_to_lidar': np.array([
            [0.707555,  -0.706598,   0.009238,  0.054002],
            [-0.004619,  -0.017697,  -0.999833, -0.007218],
            [0.706643,   0.707394,  -0.015785,  -0.088251],
            [0, 0, 0, 1]
        ], dtype=np.float64)
    }
    
    # 相机2参数
    camera_dict[2] = {
        'width': 512,
        'height': 384,
        'K': np.array([
            [504.71027203 / 2, 0, 558.67274185 / 2],
            [0, 505.94199486 / 2, 399.34953343 / 2],
            [0, 0, 1]
        ], dtype=np.float64),
        'camera_to_lidar': np.array([
            [ -0.693995,   0.719643,   0.022003,-0.005720],
            [  -0.032811,  -0.001084,  -0.999461,-0.078270],
            [  -0.719232,  -0.694343,   0.024365,-0.030108],
            [ 0, 0, 0, 1]
        ], dtype=np.float64)
    }
    
    # 相机3参数
    camera_dict[3] = {
        'width': 512,
        'height': 384,
        'K': np.array([
            [492.21619642 / 2, 0, 531.39937227 / 2],
            [0, 491.75080876 / 2, 394.35920146 / 2],
            [0, 0, 1]
        ], dtype=np.float64),
        'camera_to_lidar': np.array([
            [  0.726027,   0.687592,  -0.010122, 0.079652],
            [ 0.001577,  -0.016384,  -0.999865, 0.014322],
            [-0.687665,   0.725912,  -0.012979,-0.068272],
            [ 0, 0, 0, 1]
        ], dtype=np.float64)
    }
    
    return camera_dict

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


def project_3d_to_2d(points_3d, K, R, t):
    """将3D点投影到2D图像平面"""
    points_3d_cam = R @ points_3d.T + t.reshape(3, 1)  # 3xN
    points_3d_cam = points_3d_cam.T  # 转置为Nx3
    
    # 将3D点云投影到2D图像平面
    points_2d_homogeneous = K @ points_3d_cam.T  # 3xN
    points_2d = points_2d_homogeneous[:2, :] / points_2d_homogeneous[2, :]  # 归一化
    points_2d = points_2d.T  # 转置为Nx2
    
    return points_2d

def reverse_2d_coords(points_2d, img_width, img_height):
    """反转2D坐标的x轴和y轴（实现180度图像旋转效果）"""
    # x轴反转：图像宽度 - 原始x坐标
    points_2d[:, 0] = img_width - points_2d[:, 0]
    # y轴反转：图像高度 - 原始y坐标
    points_2d[:, 1] = img_height - points_2d[:, 1]
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

def clamp_coords(x, y, width, height):
    clamped_x = max(0, min(int(x), width))  # x坐标限制在 [0, width]
    clamped_y = max(0, min(int(y), height)) # y坐标限制在 [0, height]
    return (clamped_x, clamped_y)



def draw_cube_label(vertices_2d,image,width,height):
    edges = [
            # 前面（前右上、前右下、前左下、前左上 相连）
            (0, 2), (2, 6), (6, 4), (4, 0),
            # 后面（后右上、后右下、后左下、后左上 相连）
            (1, 3), (3, 7), (7, 5), (5, 1),
            # 连接前后对应的顶点
            (0, 1), (2, 3), (6, 7), (4, 5)
        ]

    # 按顺序绘制每条边
    for (v1, v2) in edges:   
        # 获取截断后的起点和终点
        pt1 = clamp_coords(vertices_2d[v1, 0], vertices_2d[v1, 1], width, height)
        pt2 = clamp_coords(vertices_2d[v2, 0], vertices_2d[v2, 1], width, height)
        cv2.line(
                img=image,
                pt1=pt1,
                pt2=pt2,
                color=(255, 0, 0),  # BGR格式，蓝色
                thickness=2)


        # if (0 <= vertices_2d[v1, 0] <= width and 0 <= vertices_2d[v1, 1] <= height and
        #     0 <= vertices_2d[v2, 0] <= width and 0 <= vertices_2d[v2, 1] <= height):
        #     cv2.line(
        #         img=image,
        #         pt1=[int(vertices_2d[v1, 0]), int(vertices_2d[v1, 1])],
        #         pt2=[int(vertices_2d[v2, 0]), int(vertices_2d[v2, 1])],
        #         color=(255, 0, 0),  # BGR格式，蓝色
        #         thickness=2)

        # else:
        #     break
    
    return image


def draw_rectangle_label(visible_vertices_2d,img,cube):

    bbox = compute_bounding_box(visible_vertices_2d)

    # --------------------------
    # 标签处理
    min_x_original = bbox['min_x']
    min_y_original = bbox['min_y']
    clamped_width = bbox['width']  # 原始宽（未旋转图像中）
    clamped_height = bbox['height']  # 原始高（未旋转图像中）
    max_x_original = min_x_original + clamped_width
    max_y_original = min_y_original + clamped_height


    # 3. 核心：裁剪超出图像边界的坐标（新增步骤）
    # --------------------------
    # 裁剪x坐标到[0, width-1]
    min_x_clamped = np.clip(min_x_original, 0, width - 1)
    max_x_clamped = np.clip(max_x_original, 0, width - 1)
    min_x, max_x = (min_x_clamped, max_x_clamped) if min_x_clamped <= max_x_clamped else (max_x_clamped, min_x_clamped)
    # 裁剪y坐标到[0, height-1]
    min_y_clamped = np.clip(min_y_original, 0, height - 1)
    max_y_clamped = np.clip(max_y_original, 0, height - 1)
    min_y, max_y = (min_y_clamped, max_y_clamped) if min_y_clamped <= max_y_clamped else (max_y_clamped, min_y_clamped)
    
    box_color = class_color_map.get(cube['label'], (128, 128, 128))  # 类别颜色
    class_name = cube['label']

    pt1 = (int(min_x_clamped), int(min_y_clamped))  # 裁剪后左上角
    pt2 = (int(max_x_clamped), int(max_y_clamped))  # 裁剪后右下角
    cv2.rectangle(img, pt1, pt2, box_color, 1)  # 仅绘制图像内的部分
    # 5. 绘制类别名称（适配裁剪后的位置）
    # 文本位置：裁剪后框的左上角上方（避免超出图像）
    if pt1[1] - 10 > 0:  # 上方有空间
        text_pos = (pt1[0], pt1[1] - 10)
    else:  # 上方无空间，放框内
        text_pos = (pt1[0], pt1[1] + 20)
    
    # 白色文本+黑色描边，确保清晰
    cv2.putText(img, class_name, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)  # 描边
    cv2.putText(img, class_name, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)  # 文本
    
    return pt1,pt2



def main(cubes, K, R, t,image,width,height,frame_pcd,
         label_2d_path,vis_save_dir,
         is_reverse=False):
    

    #优化方案1：提取共视区点云，判断共视区点云中是否有足够的点
    # 1.提取共视区点云
    filter_frame_pcd, _ = get_common_view_points(np.asarray(frame_pcd.points), K, R, t, width, height)
    yolo_labels = []
    vis_objects = []
    
    for cube in cubes:
        
        # 2.判断共视区点云中是否有足够的点
        bbox = o3d.geometry.OrientedBoundingBox(
                center=np.array(cube['center']),
                extent=np.array(cube['dimensions']),
                R=euler2rot(cube['rotation']),
            )
        points_translated = np.asarray(filter_frame_pcd) - bbox.center
        points_local = np.dot(points_translated, bbox.R)  # 转换到局部坐标系
        half_extent = bbox.extent / 2
        in_bbox = np.all(np.abs(points_local) <= half_extent + 1e-6, axis=1)
        num_in_box = np.sum(in_bbox)
        
        # 如果点数 >= min_points，则保留该框
        if num_in_box < min_points_list[cube['label']]:
            continue  # 点数不足，跳过该框
        
        ##------------------------------------
        # 3.计算顶点
        vertices = get_cube_vertices(
                cube['center'], 
                cube['dimensions'], 
                cube['rotation']
            )
        
        points_cam = R @ vertices.T + np.array(t).reshape(3, 1)
        visible = points_cam[2, :] > 0
        if not np.any(visible):
            continue  # 完全不可见的立方体不处理

        # 投影到2D图像平面
        vertices_2d = project_3d_to_2d(vertices, K, R, t)
        if is_reverse:# 如果图像是倒置的，需要反转2D坐标
            vertices_2d = reverse_2d_coords(vertices_2d, width, height)
        #画立体框
        draw_cube_label(vertices_2d,image,width,height)
        #画矩形框
        pt1, pt2 = draw_rectangle_label(vertices_2d,image,cube)
        # 添加到YOLO标签列表
        yolo_line = f"{cube['label']} {pt1[0]} {pt1[1]} {pt2[0]} {pt2[1]}"
        yolo_labels.append(yolo_line)

    with open(label_2d_path, 'w', encoding='utf-8') as f:
        if yolo_labels:
            f.write('\n'.join(yolo_labels) + '\n')  # 每行一个目标
        else:
            f.write('')  # 无目标则为空文件

    cv2.imwrite(vis_save_dir, image)    
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()




if __name__ == "__main__":

    # ！！！！需手动设置
    is_reverse = False # 图像是否倒置
    camera_dict = get_camera_intrinsics_dict() # 相机内参


    scene_dir = "/mnt/dln/data/datasets/0915/for_bev_test/"
    img_list = os.listdir(os.path.join(scene_dir, "Colmap"))

    label_2d_dir = scene_dir + '/2D_label'
    vis_save_dir =  scene_dir + "fixed_images_with_label"

    

    # 根据图像去遍历标签
    for direction in img_list: # 0,1,2,3
        if direction == "1":
            continue
        imgs_dir = os.path.join(scene_dir, "Colmap"+"/"+direction+"/images")
        label_2d_dir = scene_dir + "/2D_label"+"/"+direction
        vis_save_dir =  scene_dir + "/fixed_images_with_label"+"/"+direction

        if os.path.exists(vis_save_dir):
            print(f"删除已存在的可视化目录 {vis_save_dir}")
            shutil.rmtree(vis_save_dir)
        os.makedirs(vis_save_dir)
        if os.path.exists(label_2d_dir):
            print(f"删除已存在的2D标签目录 {label_2d_dir}")
            shutil.rmtree(label_2d_dir)
        os.makedirs(label_2d_dir)
    
        

        img_path_list = os.listdir(imgs_dir)
        for image_name in img_path_list:  # 00001.png, 00002.png, ...
            img_path = os.path.join(imgs_dir, image_name)
            pcd_path = os.path.join(scene_dir+"/pcd", image_name[:-4]+".pcd")  # 00001.pcd, 00002.pcd, ...
            pcd_label_path = os.path.join(scene_dir+"/3D_label", image_name[:-4]+".txt")  # 00001.txt, 00002.txt, ...
            label_2d_path = os.path.join(label_2d_dir, image_name[:-4]+".txt")
            vis_save_path = os.path.join(vis_save_dir, image_name)

            print(vis_save_path)
            cubes = parse_3d_annotations(pcd_label_path)
            frame_pcd = o3d.io.read_point_cloud(pcd_path)

            K = camera_dict[int(direction)]['K']
            R = camera_dict[int(direction)]['camera_to_lidar'][:3, :3]
            t = camera_dict[int(direction)]['camera_to_lidar'][:3, 3]
            image = cv2.imread(img_path)
            width, height = image.shape[1], image.shape[0]

            main(cubes, K, R, t,image,width,height,frame_pcd,
                 label_2d_path,vis_save_path, 
                 is_reverse)

            # print(pcd_label)
        # break
