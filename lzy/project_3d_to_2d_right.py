import open3d as o3d
import cv2
import numpy as np
import matplotlib.pyplot as plt 
import pickle
import copy

# from pcdet.utils.box_utils import boxes3d_to_corners3d_kitti_camera

def draw_projected_box3d(image, qs, color=(0, 255, 0), thickness=2):
    """ Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    """
    qs = qs.astype(np.int32)
    for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        # use LINE_AA for opencv3
        # cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

    # 新增对于朝向的可视化，不一定可靠
    cv2.line(image, (qs[2, 0], qs[2, 1]), (qs[7, 0], qs[7, 1]), color, thickness)
    cv2.line(image, (qs[3, 0], qs[3, 1]), (qs[6, 0], qs[6, 1]), color, thickness)
    return image


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


# # 读取一帧点云，投影到图像上
# pc = o3d.io.read_point_cloud('/data/datasets/3DPerception/agi_calibration/fisheye_right_to_mid360/image_0_rotate.pcd') 
pc = o3d.io.read_point_cloud('/mnt/dln/data/datasets/0915/make_label_raw/27-parking-1/pcd/1.pcd') 
# pc_np = np.asarray(pc.points) 
# angle = np.radians(-90)  # 转换为弧度
# cos_theta = np.cos(angle)
# sin_theta = np.sin(angle)
# rotation_matrix = np.array([
#     [cos_theta, -sin_theta, 0],
#     [sin_theta, cos_theta, 0],
#     [0, 0, 1]
#     ])
# pc_np[:, :3] = np.dot(pc_np[:, :3], rotation_matrix.T)
# pc_np[:, 2] = -pc_np[:, 2]

# # # (x,-y,-z+1.5)
# pc_np[:, 1] *= -1
# pc_np[:, 2] *= -1
# pc_np[:, 2] += 1.5
# pc = o3d.geometry.PointCloud()
# pc.points = o3d.utility.Vector3dVector(pc_np) 
FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
o3d.visualization.draw_geometries([pc, FOR1])

# 读取一帧未畸变矫正的鱼眼
# ori_img = cv2.imread('/data/datasets/3DPerception/agi_calibration/fisheye_right_to_mid360/image_0.png')
ori_img = cv2.imread('/mnt/dln/data/datasets/0915/make_label_raw/27-parking-3/fixed_images/00001.png')

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

# # 将点云投影到畸变后的图像
# points_homo = np.hstack([pc_np, np.ones(pc_np.shape[0], dtype=np.float32).reshape((-1, 1))])

lidar_2_cam = np.eye(4).astype(np.float32)
lidar_2_cam [:3, :3] = np.array([
    [ -0.488561,   0.872221,  -0.023225],
    [0.056886,   0.005281,  -0.998367],
    [-0.870674,  -0.489084,  -0.052197]
])
lidar_2_cam[:3, 3] = np.array([0.015570,   1.497755,  -0.023615])

# # 1. 逆时针旋转 90 度的变换矩阵
# angle = np.radians(-90)
# cos_theta = np.cos(angle)
# sin_theta = np.sin(angle)
# rotation_matrix = np.array([
#     [cos_theta, -sin_theta, 0, 0],
#     [sin_theta, cos_theta, 0, 0],
#     [0, 0, 1, 0],
#     [0, 0, 0, 1]
# ], dtype=np.float32)

# # 2. z 轴取反的变换矩阵
# z_flip_matrix = np.array([
#     [1, 0, 0, 0],
#     [0, 1, 0, 0],
#     [0, 0, -1, 0],
#     [0, 0, 0, 1]
# ], dtype=np.float32)

# # 3. y 轴取反的变换矩阵
# y_flip_matrix = np.array([
#     [1, 0, 0, 0],
#     [0, -1, 0, 0],
#     [0, 0, 1, 0],
#     [0, 0, 0, 1]
# ], dtype=np.float32)

# # 4. z 轴取反并加 1.5 的变换矩阵
# z_flip_and_shift_matrix = np.array([
#     [1, 0, 0, 0],
#     [0, 1, 0, 0],
#     [0, 0, -1, 1.5],
#     [0, 0, 0, 1]
# ], dtype=np.float32)

# # 5. 合并所有变换矩阵
# total_transform_matrix = rotation_matrix @ z_flip_matrix @ y_flip_matrix @ z_flip_and_shift_matrix

# # 6. 计算新的标定矩阵
# new_lidar_2_cam = lidar_2_cam @ total_transform_matrix
# print (new_lidar_2_cam)
# pc_cam_homo = points_homo @ lidar_2_cam.T  # 形状[N,4] (N为点数)
# points_cam = pc_cam_homo[:, :3]  # 相机坐标系X,Y,Z（非齐次）
# mask = points_cam[:, 2] > 0  # 保留Z>0的有效点
# points_cam = points_cam[mask]
# points_norm = points_cam[:, :2] / points_cam[:, 2:3] 
# # 应用内参矩阵得到图像坐标（u=fx*x + cx, v=fy*y + cy）
# points_img = np.dot(points_norm, K[:2, :2].T) + K[:2, 2]  # 形状[M,2]
# # 新增：将投影点绘制到矫正后的图像上
# # 1. 过滤越界点（确保u在[0, 1600), v在[0, 1200)）
# h, w = undistorted_img.shape[:2]  # 获取图像尺寸（高1200，宽1600）
# new_undistorted_img = undistorted_img.copy()

# valid_mask = (points_img[:, 0] >= 0) & (points_img[:, 0] < w) & \
#              (points_img[:, 1] >= 0) & (points_img[:, 1] < h)
# points_img_valid = points_img[valid_mask].astype(np.int32)  # 转换为整数像素坐标

# # 2. 绘制散点（红色，半径2，厚度-1表示填充）
# for (u, v) in points_img_valid:
#     cv2.circle(new_undistorted_img, (u, v), radius=2, color=(0, 0, 255), thickness=-1)

# # 3. 显示绘制后的图像
# cv2.imshow("undistorted_img_with_points", new_undistorted_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # 因此，原始雷达坐标系下的点需要通过两次过滤：mask（Z>0）和valid_mask（越界）
# valid_3d_points = pc_np[mask][valid_mask]  # 原始雷达坐标系下的有效3D点

# # 2. 提取投影点的颜色（BGR转RGB）
# colors = []
# for (u, v) in points_img_valid:
#     # 读取图像对应位置的BGR颜色（注意OpenCV图像坐标是(v, u)）
#     bgr_color = undistorted_img[v, u]
#     rgb_color = (bgr_color[2]/255, bgr_color[1]/255, bgr_color[0]/255)  # 转换为0-1的RGB值
#     print (rgb_color)
#     colors.append(rgb_color)

# # 3. 创建带颜色的Open3D点云
# colored_pc = o3d.geometry.PointCloud()
# colored_pc.points = o3d.utility.Vector3dVector(valid_3d_points)  # 3D坐标
# colored_pc.colors = o3d.utility.Vector3dVector(colors)  # 颜色（0-1范围的RGB）

# # 4. 可视化带颜色的点云（添加坐标系辅助）
# FOR = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
# o3d.visualization.draw_geometries([colored_pc, FOR])


# 加载对应的检测结果
pkl_path = '/data/project/OpenPCDet_ziyu/output/15b_models/centerpoint_ped_15b/use3000/eval/eval_with_train/epoch_79/val/result.pkl'
with open(pkl_path, 'rb') as f:
    infos = pickle.load(f)  # 加载pkl数据
info = infos[0]
# import pdb; pdb.set_trace()
boxes_3d = info['boxes_lidar']
scores = info['score']

# 模型结果对应到原始坐标系中
angle = np.radians(-90)  # 转换为弧度
cos_theta = np.cos(angle)
sin_theta = np.sin(angle)
rotation_matrix = np.array([
    [cos_theta, -sin_theta, 0],
    [sin_theta, cos_theta, 0],
    [0, 0, 1]
    ])
boxes_3d[:, :3] = np.dot(boxes_3d[:, :3], rotation_matrix.T)
boxes_3d[:, 6] += np.pi/2
boxes_3d[:, 2] = -boxes_3d[:, 2]

# 暂时只保留x<0的结果，因为是智元用
mask = (boxes_3d[:, 0] < 0)
boxes_3d = boxes_3d[mask]
# # (x,-y,-z+1.5)
boxes_3d[:, 1] *= -1
boxes_3d[:, 2] *= -1
boxes_3d[:, 2] += 1.5

# 注意，此处的朝向角度需要再次检查

# 将3d检测结果转到相机坐标下
boxes_3d_copy = copy.deepcopy(boxes_3d)
xyz_lidar = boxes_3d_copy[:, 0:3]
l, w, h = boxes_3d_copy[:, 3:4], boxes_3d_copy[:, 4:5], boxes_3d_copy[:, 5:6]
r = boxes_3d_copy[:, 6:7]

# 移动到bottom center，后面转corners3d要用
xyz_lidar[:, 2] -= h.reshape(-1) / 2

# 将点云投影到畸变后的图像
xyz_lidar_homo = np.hstack([xyz_lidar, np.ones(xyz_lidar.shape[0], dtype=np.float32).reshape((-1, 1))])

xyz_cam_homo = xyz_lidar_homo @ lidar_2_cam.T  # 形状[N,4] (N为点数)
xyz_cam = xyz_cam_homo[:, :3]  # 相机坐标系X,Y,Z（非齐次）
# xyz_cam[:, 1] += h.reshape(-1) / 2
r = -r - np.pi / 2
boxes_cam = np.concatenate([xyz_cam, l, h, w, r], axis=-1)
boxes_cam_corners3d = boxes3d_to_corners3d_kitti_camera(boxes_cam).reshape(-1, 3)


# mask = boxes_cam_corners3d[:, 2] > 0  # 保留Z>0的有效点
# boxes_cam_corners3d = boxes_cam_corners3d[mask]
boxes_cam_corners3d_norm = boxes_cam_corners3d[:, :2] / boxes_cam_corners3d[:, 2:3] 
# 应用内参矩阵得到图像坐标（u=fx*x + cx, v=fy*y + cy）
boxes_img_corners3d = np.dot(boxes_cam_corners3d_norm, K[:2, :2].T) + K[:2, 2]  # 形状[M,2]
boxes_img_corners3d = boxes_img_corners3d.astype(np.int32).reshape(-1, 8, 2)  # 转换为整数像素坐标

min_uv = np.min(boxes_img_corners3d, axis=1)  # (N, 2)
max_uv = np.max(boxes_img_corners3d, axis=1)  # (N, 2)
boxes2d_image = np.concatenate([min_uv, max_uv], axis=1)
# boxes2d_image[:, 0] = np.clip(boxes2d_image[:, 0], a_min=0, a_max=640 - 1)
# boxes2d_image[:, 1] = np.clip(boxes2d_image[:, 1], a_min=0, a_max=480 - 1)
# boxes2d_image[:, 2] = np.clip(boxes2d_image[:, 2], a_min=0, a_max=640 - 1)
# boxes2d_image[:, 3] = np.clip(boxes2d_image[:, 3], a_min=0, a_max=480 - 1)

print (boxes2d_image)
for i in range(boxes2d_image.shape[0]):
    if boxes2d_image[i, 0] >= 0 and boxes2d_image[i, 0] <= 639 and boxes2d_image[i, 1] >= 0 and boxes2d_image[i, 1] <= 479 and boxes2d_image[i, 2] >= 0 and boxes2d_image[i, 2] <= 639  and boxes2d_image[i, 3] >= 0 and boxes2d_image[i, 3] <= 479:
        # cv2.rectangle(
        #             undistorted_img,
        #             (int(boxes2d_image[i, 0]), int(boxes2d_image[i, 1])),
        #             (int(boxes2d_image[i, 2]), int(boxes2d_image[i, 3])),
        #             (255, 255, 0),
        #             2,
        #         )
        draw_projected_box3d(undistorted_img, boxes_img_corners3d[i, ...])

# 3. 显示绘制后的图像
cv2.imshow("3d project result", undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()