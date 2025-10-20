import pandas as pd
import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R  # 用于更可靠的欧拉角转换


# --------------------------
# 基础工具函数
# --------------------------
def read_pcd(pcd_path):
    pcd = o3d.io.read_point_cloud(pcd_path)
    if pcd.is_empty():
        raise ValueError("点云文件读取失败，请检查路径！")
    return np.asarray(pcd.points), pcd

def read_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("图像文件读取失败，请检查路径！")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def read_labels(label_path):
    labels = pd.read_csv(label_path, sep=',')
    print(f"原始标签数量：{len(labels)}")
    return labels

def save_labels(labels, save_path):
    labels.to_csv(save_path, index=False)
    print(f"筛选后的标签已保存至：{save_path}")


# --------------------------
# 坐标转换与共视区提取
# --------------------------
def get_lidar_to_cam_extrinsics(camera_to_lidar):
    """
    从相机到雷达的外参矩阵，计算雷达到相机的外参（R, T）
    camera_to_lidar: 4x4 矩阵（相机→雷达的变换）
    返回：
        R: 3x3 旋转矩阵（雷达→相机）
        T: 3x1 平移向量（雷达→相机）
    """
    # 计算相机到雷达矩阵的逆，得到雷达→相机的变换
    lidar_to_cam = np.linalg.inv(camera_to_lidar)
    R = lidar_to_cam[:3, :3]  # 旋转矩阵
    T = lidar_to_cam[:3, 3]   # 平移向量
    return R, T

def euler_to_rotation_matrix(euler_deg):
    """将欧拉角（度，Z-Y-X顺序）转换为旋转矩阵"""
    r = R.from_euler('xyz', euler_deg, degrees=True)
    return r.as_matrix()

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


# --------------------------
# 标签筛选与可视化
# --------------------------
def is_object_in_common_view(center, extent, rotation, K, R_lidar2cam, T_lidar2cam, img_width, img_height):
    """判断物体是否在共视区（基于包围盒顶点）"""
    half_extent = np.array(extent) / 2
    # 生成8个顶点（局部坐标系）
    vertices_local = np.array([
        [half_extent[0], half_extent[1], half_extent[2]],
        [half_extent[0], half_extent[1], -half_extent[2]],
        [half_extent[0], -half_extent[1], half_extent[2]],
        [half_extent[0], -half_extent[1], -half_extent[2]],
        [-half_extent[0], half_extent[1], half_extent[2]],
        [-half_extent[0], half_extent[1], -half_extent[2]],
        [-half_extent[0], -half_extent[1], half_extent[2]],
        [-half_extent[0], -half_extent[1], -half_extent[2]]
    ])
    
    # 顶点转换到雷达坐标系（应用物体旋转和中心偏移）
    R_obj = euler_to_rotation_matrix(rotation)  # 物体旋转矩阵
    vertices_lidar = (R_obj @ vertices_local.T).T + center  # (8,3)
    
    # 转换到相机坐标系
    vertices_hom = np.hstack([vertices_lidar, np.ones((8,1))])  # (8,4)
    extrinsic = np.hstack([R_lidar2cam, T_lidar2cam.reshape(3,1)])
    vertices_cam = (extrinsic @ vertices_hom.T).T  # (8,3)
    
    # 筛选相机前方的顶点
    front_vertices = vertices_cam[vertices_cam[:, 2] > 1e-6]
    if len(front_vertices) == 0:
        return False
    
    # 投影到图像
    Z = front_vertices[:, 2].reshape(-1, 1)
    X = front_vertices[:, 0].reshape(-1, 1)
    Y = front_vertices[:, 1].reshape(-1, 1)
    u = (K[0,0] * X / Z + K[0,2]).flatten()
    v = (K[1,1] * Y / Z + K[1,2]).flatten()
    
    # 检查是否有顶点在图像内
    in_img = (u >= 0) & (u < img_width) & (v >= 0) & (v < img_height)
    return np.any(in_img)

def filter_labels(labels, K, R_lidar2cam, T_lidar2cam, img_width, img_height):
    """筛选共视区标签"""
    mask = []
    for _, row in labels.iterrows():
        center = np.array([row['center_x'], row['center_y'], row['center_z']])
        extent = np.array([row['extent_x'], row['extent_y'], row['extent_z']])
        rotation = np.array([row['R_x'], row['R_y'], row['R_z']])  # Z-Y-X欧拉角（度）
        mask.append(is_object_in_common_view(center, extent, rotation, K, R_lidar2cam, T_lidar2cam, img_width, img_height))
    filtered = labels[mask].reset_index(drop=True)
    print(f"共视区标签数量：{len(filtered)}")
    return filtered

def create_3d_bbox(center, extent, rotation):
    """创建3D有向包围盒（Open3D格式）"""
    half_extent = np.array(extent) / 2
    R_obj = euler_to_rotation_matrix(rotation)  # 物体旋转矩阵
    bbox = o3d.geometry.OrientedBoundingBox(
        center=center,
        R=R_obj,
        extent=extent  # Open3D的extent是全尺寸，不是半尺寸
    )
    bbox.color = [0, 1, 0]  # 绿色包围盒
    return bbox

def project_3d_bbox_to_2d(bbox_vertices_lidar, K, R_lidar2cam, T_lidar2cam):
    """将3D包围盒顶点投影到2D图像，返回2D bbox坐标"""
    # 转换顶点到相机坐标系
    vertices_hom = np.hstack([bbox_vertices_lidar, np.ones((8,1))])
    extrinsic = np.hstack([R_lidar2cam, T_lidar2cam.reshape(3,1)])
    vertices_cam = (extrinsic @ vertices_hom.T).T  # (8,3)
    
    # 筛选相机前方的顶点并投影
    front_mask = vertices_cam[:, 2] > 1e-6
    vertices_cam_front = vertices_cam[front_mask]
    if len(vertices_cam_front) == 0:
        return None  # 无有效顶点
    
    Z = vertices_cam_front[:, 2].reshape(-1, 1)
    X = vertices_cam_front[:, 0].reshape(-1, 1)
    Y = vertices_cam_front[:, 1].reshape(-1, 1)
    u = (K[0,0] * X / Z + K[0,2]).flatten()
    v = (K[1,1] * Y / Z + K[1,2]).flatten()
    
    # 计算2D bbox（最小外接矩形）
    u_min, u_max = np.min(u), np.max(u)
    v_min, v_max = np.min(v), np.max(v)
    return np.array([u_min, v_min, u_max, v_max]).astype(int)  # [x1, y1, x2, y2]

def visualize_labels(common_pcd, filtered_labels, img, uvs, K, R_lidar2cam, T_lidar2cam):
    """可视化共视区点云+3D包围盒，以及图像+2D bbox"""
    # 1. 3D可视化：共视区点云 + 标签包围盒
    geometries = [common_pcd]  # 红色共视区点云
    for _, row in filtered_labels.iterrows():
        center = np.array([row['center_x'], row['center_y'], row['center_z']])
        extent = np.array([row['extent_x'], row['extent_y'], row['extent_z']])
        rotation = np.array([row['R_x'], row['R_y'], row['R_z']])

        bbox = create_3d_bbox(center, extent, rotation)
        geometries.append(bbox)
    o3d.visualization.draw_geometries(geometries, window_name="共视区点云+3D包围盒")
    
    # 2. 2D可视化：图像 + 点云投影 + 2D bbox
    img_copy = img.copy()
    # 绘制点云投影
    for u, v in uvs:
        cv2.circle(img_copy, (u, v), 1, (255, 0, 0), -1)  # 红色点
    
    # 绘制2D bbox和标签
    for _, row in filtered_labels.iterrows():
        # 生成3D包围盒顶点（雷达坐标系）
        center = np.array([row['center_x'], row['center_y'], row['center_z']])
        extent = np.array([row['extent_x'], row['extent_y'], row['extent_z']])
        rotation = np.array([row['R_x'], row['R_y'], row['R_z']])
        half_extent = extent / 2
        vertices_local = np.array([
            [half_extent[0], half_extent[1], half_extent[2]],
            [half_extent[0], half_extent[1], -half_extent[2]],
            [half_extent[0], -half_extent[1], half_extent[2]],
            [half_extent[0], -half_extent[1], -half_extent[2]],
            [-half_extent[0], half_extent[1], half_extent[2]],
            [-half_extent[0], half_extent[1], -half_extent[2]],
            [-half_extent[0], -half_extent[1], half_extent[2]],
            [-half_extent[0], -half_extent[1], -half_extent[2]]
        ])
        R_obj = euler_to_rotation_matrix(rotation)
        vertices_lidar = (R_obj @ vertices_local.T).T + center
        
        # 投影到2D并绘制
        bbox_2d = project_3d_bbox_to_2d(vertices_lidar, K, R_lidar2cam, T_lidar2cam)
        if bbox_2d is not None:
            x1, y1, x2, y2 = bbox_2d
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色框
            cv2.putText(
                img_copy, row['label'], (x1, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )  # 类别标签
    
    # 显示图像
    plt.figure(figsize=(10, 8))
    plt.imshow(img_copy)
    plt.title("图像+点云投影+2D包围盒")
    plt.axis('off')
    plt.show()


# --------------------------
# 主流程
# --------------------------
if __name__ == "__main__":
    # --------------------------
    # 配置参数（替换为你的实际数据）
    # --------------------------
    pcd_path = "/mnt/dln/data/datasets/0915/make_label_raw/27-parking-1/pcd/759.pcd"
    img_path = "/mnt/dln/data/datasets/0915/make_label_raw/27-parking-1/fixed_images/00759.png"
    label_path = "/mnt/dln/data/datasets/0915/make_label_raw/27-parking-1/3D_label/759.txt"
    save_label_path = "/mnt/dln/data/datasets/0915/make_label_raw/27-parking-1/test_filtered_759.txt"
    
    # 相机内参
    cam_fx = 605.231000
    cam_fy = 605.133900
    cam_cx = 320.931700
    cam_cy = 253.091400
    width = 640  # 图像宽度
    height = 480  # 图像高度
    K = np.array([
        [cam_fx, 0, cam_cx],
        [0, cam_fy, cam_cy],
        [0, 0, 1]
    ])
    
    # 相机到雷达的外参矩阵（4x4）
    camera_to_lidar = np.array([
        [-0.028191, -0.9996, 0.000532, 0.033057],
        [0.430632, -0.012625, -0.902439, -0.116977],
        [0.902087, -0.025211, 0.430817, 0.013525],
        [0, 0, 0, 1]
    ], dtype=np.float64)
    
    # 计算雷达到相机的外参（R, T）
    # R_lidar2cam, T_lidar2cam = get_lidar_to_cam_extrinsics(camera_to_lidar)

    R_lidar2cam = camera_to_lidar[:3, :3]
    T_lidar2cam = camera_to_lidar[:3, 3]
    # --------------------------
    # 执行流程
    # --------------------------
    # 1. 读取数据
    points, original_pcd = read_pcd(pcd_path)
    img = read_image(img_path)
    labels = read_labels(label_path)
    
    # 2. 提取共视区点云
    common_points, uvs = get_common_view_points(
        points, K, R_lidar2cam, T_lidar2cam, width, height
    )
    
    # 3. 筛选共视区标签
    filtered_labels = filter_labels(
        labels, K, R_lidar2cam, T_lidar2cam, width, height
    )
    
    # 4. 保存筛选后的标签
    save_labels(filtered_labels, save_label_path)
    
    # 5. 可视化（共视区点云+3D包围盒，图像+2D包围盒）
    common_pcd = o3d.geometry.PointCloud()
    common_pcd.points = o3d.utility.Vector3dVector(common_points)
    common_pcd.paint_uniform_color([1, 0, 0])  # 共视区点云标为红色
    visualize_labels(common_pcd, filtered_labels, img, uvs, K, R_lidar2cam, T_lidar2cam)
