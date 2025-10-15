import pandas as pd
import math
import numpy as np
import cv2

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


labels = '/mnt/dln/data/datasets/0915/make_label_raw/lab_433/3D_label/1002.txt'
cubes = parse_3d_annotations(labels)

img_path = '/mnt/dln/data/datasets/0915/make_label_raw/lab_433/fixed_images/01002.png'
image = cv2.imread(img_path)


fx = 605.231000
fy = 605.133900  # 焦距y
cx = 320.931700   # 主点x坐标
cy = 253.091400   # 主点y坐标
width = 640 # 图像宽度
height = 480 # 图像高度
# 内参矩阵
K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
])

camera_to_lidar = np.array([
    [-0.028191, -0.9996, 0.000532, 0.033057],
    [0.430632, -0.012625, -0.902439, -0.116977],
    [0.902087, -0.025211, 0.430817, 0.013525],
    [0, 0, 0, 1]
    ], dtype=np.float64)
# lidar_to_camera = np.linalg.inv(camera_to_lidar)


for cube in cubes:
    vertices = get_cube_vertices(cube['center'], cube['dimensions'], cube['rotation'])
    
#     #  将3D点云从世界坐标系转换到相机坐标系
    points_3d_cam = camera_to_lidar[:3,:3] @ vertices.T + np.array(camera_to_lidar[:3, 3]).reshape(3, 1)  # 3xN
    points_3d_cam = points_3d_cam.T  # 转置为Nx3
    
    # 将3D点云投影到2D图像平面
    points_2d_homogeneous = K @ points_3d_cam.T  # 3xN
    points_2d = points_2d_homogeneous[:2, :] / points_2d_homogeneous[2, :]  # 归一化
    points_2d = points_2d.T  # 转置为Nx2

#     # 将2D点绘制到图像上
#     for point in points_2d:
#         x, y = int(point[0]), int(point[1])
#         if 0 <= x < width and 0 <= y < height:  # 确保点在图像范围内
#             cv2.circle(image, (x, y), 3, (0, 255, 0), -1)  # 绘制绿色圆点
 
# # 显示图像
# cv2.imshow("2D Projection of Point Cloud", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


    # 过滤掉在相机后方的点 (z < 0)
    # points_cam = camera_to_lidar[:3,:3] @ vertices.T + np.array(camera_to_lidar[:3, 3]).reshape(3, 1)
    # visible = points_cam[2, :] > 0

    #  # 获取可见的2D顶点
    # visible_vertices_2d = vertices[visible]
    bbox = compute_bounding_box(points_2d)


    min_x_original = bbox['min_x']
    min_y_original = bbox['min_y']
    clamped_width = bbox['width']  # 原始宽（未旋转图像中）
    clamped_height = bbox['height']  # 原始高（未旋转图像中）
    max_x_original = min_x_original + clamped_width
    max_y_original = min_y_original + clamped_height
    
    # 2. 计算原始框面积，防护面积为0的异常情况
    original_area = clamped_width * clamped_height
    if original_area <= 0:
        continue  # 原始框无效，跳过

    # 2. 转换坐标到180度旋转后的图像
    is_rotated_180 = True  # 图像是否旋转了180度
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
    # 裁剪y坐标到[0, height-1]
    min_y_clamped = np.clip(min_y_rot, 0, height - 1)
    max_y_clamped = np.clip(max_y_rot, 0, height - 1)
    
    # 计算裁剪后的有效宽高，过滤无效框
        # 5. 计算有效框面积及占比，过滤小占比框（核心优化）
    valid_width = max_x_clamped - min_x_clamped
    valid_height = max_y_clamped - min_y_clamped
    valid_area = valid_width * valid_height
    area_ratio = valid_area / original_area  # 有效面积占比

    # 框占图像面积占比
    area = width  * height
    ratio = valid_area / area

    MIN_AREA_RATIO = 0.5  # 最小有效面积占比阈值
    # 若有效面积为0 或 占比＜1/5，丢弃该框
    flag = area_ratio < MIN_AREA_RATIO
    # flag = area_ratio < MIN_AREA_RATIO or ratio < MIN_AREA_RATIO
    if valid_area <= 0 or flag:
        # print(f"丢弃小占比框：有效占比={area_ratio:.3f}<{MIN_AREA_RATIO})")
        continue
    
    # 4. 绘制带颜色的矩形框（使用裁剪后的坐标）
    box_color =  (128, 128, 128)  # 类别颜色
    class_name = cube['label']

    pt1 = (int(min_x_clamped), int(min_y_clamped))  # 裁剪后左上角
    pt2 = (int(max_x_clamped), int(max_y_clamped))  # 裁剪后右下角
    cv2.rectangle(image, pt1, pt2, box_color, 2)  # 仅绘制图像内的部分
    cv2.putText(image, class_name, (pt1[0], pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    
   
cv2.imshow('3D Bounding Boxes', image)
cv2.waitKey(0)  # 等待按键关闭窗口  
cv2.destroyAllWindows()
# cv2.imwrite('/mnt/dln/data/datasets/0915/make_label_raw/27-parking-1/2d_imgs_test/output_labeled_image.jpg', image)
