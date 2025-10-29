import cv2
import numpy as np
import pandas as pd
import math


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

def main(cubes, K, R, t,image,width,height):
    for cube in cubes:
        vertices = get_cube_vertices(
                cube['center'], 
                cube['dimensions'], 
                cube['rotation']
            )
        
        points_cam = R @ vertices.T + np.array(t).reshape(3, 1)
        visible = points_cam[2, :] > 0
        if not np.any(visible):
            # print("continue")
            continue  # 完全不可见的立方体不处理


        vertices_2d = project_3d_to_2d(vertices, K, R, t)
        vertices_2d = reverse_2d_coords(vertices_2d, width, height)
        
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
            # 从顶点v1到v2画一条蓝色线条（厚度2）
            # print(vertices_2d[v1],vertices_2d[v2])
            
            # vertices_2d[v1, 0] = max(0, min(width, vertices_2d[v1, 0]))
            # vertices_2d[v1, 1] = max(0, min(height, vertices_2d[v1, 1]))
            # vertices_2d[v2, 0] = max(0, min(width, vertices_2d[v2, 0]))
            # vertices_2d[v2, 1] = max(0, min(height, vertices_2d[v2, 1]))
            if (0 <= vertices_2d[v1, 0] <= width and 0 <= vertices_2d[v1, 1] <= height and
                    0 <= vertices_2d[v2, 0] <= width and 0 <= vertices_2d[v2, 1] <= height):

                cv2.line(
                img=image,
                pt1=[int(vertices_2d[v1, 0]), int(vertices_2d[v1, 1])],
                pt2=[int(vertices_2d[v2, 0]), int(vertices_2d[v2, 1])],
                color=(255, 0, 0),  # BGR格式，蓝色
                thickness=2)
            else:
                break
        
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    scence_dir = "/mnt/dln/data/datasets/0915/make_label_raw/lab_431/"
    file = "00810"
    txt_path = scence_dir + "/3D_label/" + file[-3:] + ".txt"
    img_path = scence_dir + "/fixed_images/" + file + ".png"


    cubes = parse_3d_annotations(txt_path)

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
    R = camera_to_lidar[:3, :3]
    t = camera_to_lidar[:3, 3]

     # 内参矩阵
    K = np.array([
        [cam_fx, 0, cam_cx],
        [0, cam_fy, cam_cy],
        [0, 0, 1]
    ])

    image = cv2.imread(img_path)
    main(cubes, K, R, t,image,width,height)