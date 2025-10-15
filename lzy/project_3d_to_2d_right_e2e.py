import open3d as o3d
import cv2
import numpy as np
import matplotlib.pyplot as plt 
import pickle
import copy

from pcdet.utils.box_utils import boxes3d_to_corners3d_kitti_camera

def fisheye_undistort(image, K, D):
    """
    
    参数:
        image: 输入的畸变图像 (numpy数组)
        K: 相机内参矩阵
        D: 畸变系数 [k1, k2, k3, k4]
        DIM: 图像尺寸 (width, height)

    """
    h, w = image.shape[:2]
    # 构建新的相机矩阵，保持与原图像相同的尺寸
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K, (w, h), cv2.CV_16SC2)
    
    # 使用映射进行畸变矫正
    undistorted_img = cv2.remap(
        image, map1, map2, 
        interpolation=cv2.INTER_LINEAR)
    
    return undistorted_img


# 读取一帧点云，投影到图像上 
pc = o3d.io.read_point_cloud('/data/datasets/3DPerception/agi_433/raw_data/velodyne/2025-05-16 15_36_27.072595-64551.pcd') 
pc_np = np.asarray(pc.points) 
# 将点云转换为齐次坐标，形状为 [N, 4]
points_homo = np.hstack([pc_np, np.ones((pc_np.shape[0], 1), dtype=np.float32)])


# 1. 构建绕 z 轴逆时针旋转 -90 度的 4x4 齐次变换矩阵
angle = np.radians(-90)
cos_theta = np.cos(angle)
sin_theta = np.sin(angle)
rotation_matrix = np.array([
    [cos_theta, -sin_theta, 0, 0],
    [sin_theta, cos_theta, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], dtype=np.float32)

# 2. 构建 y 轴取反的 4x4 齐次变换矩阵
y_flip_matrix = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], dtype=np.float32)

# 3. 构建 z 轴平移 1.5 米的 4x4 齐次变换矩阵
z_shift_matrix = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 1.5],
    [0, 0, 0, 1]
], dtype=np.float32)


# 读取一帧未畸变矫正的鱼眼
# ori_img = cv2.imread('/data/datasets/3DPerception/agi_calibration/fisheye_right_to_mid360/image_0.png')
ori_img = cv2.imread('/data/datasets/3DPerception/agi_433/raw_data/image1/2025-05-16 15_36_27(1).072595-64551(1).png')

ori_img = cv2.rotate(ori_img, cv2.ROTATE_180) 
cv2.imshow("original fisheye", ori_img)
cv2.waitKey(0)  # 等待任意按键
cv2.destroyAllWindows()  

# 为鱼眼图像进行畸变矫正
# 相机内参矩阵
# K = [fx, 0, cx]
#     [0, fy, cy]
#     [0, 0,   1]
K = np.array([
    [198.596, 0, 319.124],
    [0, 198.733, 241.141],
    [0, 0, 1]
], dtype=np.float64)

# 鱼眼畸变系数 [k1, k2, k3, k4]
# 使用你提供的前4个系数，第5个忽略
D = np.array([0.03310900445189785, -0.010238211257508264, -0.0028670801037428834, -0.0013280191965369754], 
            dtype=np.float64)

undistorted_img = fisheye_undistort(ori_img, K, D)

cv2.imshow("undistorted fisheye", undistorted_img)
cv2.waitKey(0)  # 等待任意按键
cv2.destroyAllWindows()  

lidar_2_cam = np.eye(4).astype(np.float32)
lidar_2_cam [:3, :3] = np.array([
    [ -0.488561,   0.872221,  -0.023225],
    [0.056886,   0.005281,  -0.998367],
    [-0.870674,  -0.489084,  -0.052197]
])
lidar_2_cam[:3, 3] = np.array([0.015570,   1.497755,  -0.023615])

# new_lidar_2_cam = (rotation_matrix.T @ y_flip_matrix.T @ z_shift_matrix.T @ lidar_2_cam.T).T
new_lidar_2_cam2 = np.array([
    [8.7222099e-01, -4.8856100e-01, -2.3225000e-02, -1.9267499e-02],
    [5.2809999e-03,  5.6885999e-02, -9.9836701e-01,  2.0456314e-04],
    [-4.8908401e-01, -8.7067401e-01, -5.2196998e-02, -1.0191050e-01],
    [0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]
], dtype=np.float32)
# 打印数据类型
pc_cam_homo = points_homo @ new_lidar_2_cam2.T  # 形状[N,4] (N为点数)
points_cam = pc_cam_homo[:, :3]  # 相机坐标系X,Y,Z（非齐次）
mask = points_cam[:, 2] > 0  # 保留Z>0的有效点
points_cam = points_cam[mask]
points_norm = points_cam[:, :2] / points_cam[:, 2:3] 
# 应用内参矩阵得到图像坐标（u=fx*x + cx, v=fy*y + cy）
points_img = np.dot(points_norm, K[:2, :2].T) + K[:2, 2]  # 形状[M,2]
# 新增：将投影点绘制到矫正后的图像上
# 1. 过滤越界点（确保u在[0, 1600), v在[0, 1200)）
h, w = undistorted_img.shape[:2]  # 获取图像尺寸（高1200，宽1600）
new_undistorted_img = undistorted_img.copy()

valid_mask = (points_img[:, 0] >= 0) & (points_img[:, 0] < w) & \
             (points_img[:, 1] >= 0) & (points_img[:, 1] < h)
points_img_valid = points_img[valid_mask].astype(np.int32)  # 转换为整数像素坐标

# 2. 绘制散点（红色，半径2，厚度-1表示填充）
for (u, v) in points_img_valid:
    cv2.circle(new_undistorted_img, (u, v), radius=2, color=(0, 0, 255), thickness=-1)

# 3. 显示绘制后的图像
cv2.imshow("undistorted_img_with_points", new_undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 因此，原始雷达坐标系下的点需要通过两次过滤：mask（Z>0）和valid_mask（越界）
valid_3d_points = pc_np[mask][valid_mask]  # 原始雷达坐标系下的有效3D点

# 2. 提取投影点的颜色（BGR转RGB）
colors = []
for (u, v) in points_img_valid:
    # 读取图像对应位置的BGR颜色（注意OpenCV图像坐标是(v, u)）
    bgr_color = undistorted_img[v, u]
    rgb_color = (bgr_color[2]/255, bgr_color[1]/255, bgr_color[0]/255)  # 转换为0-1的RGB值
    # print (rgb_color)
    colors.append(rgb_color)

# 3. 创建带颜色的Open3D点云
colored_pc = o3d.geometry.PointCloud()
colored_pc.points = o3d.utility.Vector3dVector(valid_3d_points)  # 3D坐标
colored_pc.colors = o3d.utility.Vector3dVector(colors)  # 颜色（0-1范围的RGB）

# 4. 可视化带颜色的点云（添加坐标系辅助）
FOR = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
o3d.visualization.draw_geometries([colored_pc, FOR])


# 结论：智元老头采集的训练点云，经过这个lidar2cam，可以直接投影到右鱼眼上，包括了从姜工坐标系到标定坐标系再到相机
new_lidar_2_cam2 = np.array([
    [8.7222099e-01, -4.8856100e-01, -2.3225000e-02, -1.9267499e-02],
    [5.2809999e-03,  5.6885999e-02, -9.9836701e-01,  2.0456314e-04],
    [-4.8908401e-01, -8.7067401e-01, -5.2196998e-02, -1.0191050e-01],
    [0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]
], dtype=np.float32)

# 所以相机到雷达的矩阵就是它的逆矩阵
new_cam_2_lidar = np.linalg.inv(new_lidar_2_cam2)
print (new_cam_2_lidar)
# new_cam_2_lidar = np.array([
#     [8.7222099e-01, -4.8856100e-01, -2.3225000e-02, -1.9267499e-02],
#     [5.2809999e-03,  5.6885999e-02, -9.9836701e-01,  2.0456314e-04],
#     [-4.8908401e-01, -8.7067401e-01, -5.2196998e-02, -1.0191050e-01],
#     [0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]
# ])
