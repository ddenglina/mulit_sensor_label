import open3d as o3d
import cv2
import numpy as np
import matplotlib.pyplot as plt 
import pickle
import copy

def boxes3d_to_corners3d_kitti_camera(boxes3d, bottom_center=True):
    """
    :param boxes3d: (N, 7) [x, y, z, l, h, w, ry] in camera coords, see the definition of ry in KITTI dataset
    :param bottom_center: whether y is on the bottom center of object
    :return: corners3d: (N, 8, 3)
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    """
    boxes_num = boxes3d.shape[0]
    l, h, w = boxes3d[:, 3], boxes3d[:, 4], boxes3d[:, 5]
    x_corners = np.array([l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2], dtype=np.float32).T
    z_corners = np.array([w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.], dtype=np.float32).T
    if bottom_center:
        y_corners = np.zeros((boxes_num, 8), dtype=np.float32)
        y_corners[:, 4:8] = -h.reshape(boxes_num, 1).repeat(4, axis=1)  # (N, 8)
    else:
        y_corners = np.array([h / 2., h / 2., h / 2., h / 2., -h / 2., -h / 2., -h / 2., -h / 2.], dtype=np.float32).T

    ry = boxes3d[:, 6]
    zeros, ones = np.zeros(ry.size, dtype=np.float32), np.ones(ry.size, dtype=np.float32)
    rot_list = np.array([[np.cos(ry), zeros, -np.sin(ry)],
                         [zeros, ones, zeros],
                         [np.sin(ry), zeros, np.cos(ry)]])  # (3, 3, N)
    R_list = np.transpose(rot_list, (2, 0, 1))  # (N, 3, 3)

    temp_corners = np.concatenate((x_corners.reshape(-1, 8, 1), y_corners.reshape(-1, 8, 1),
                                   z_corners.reshape(-1, 8, 1)), axis=2)  # (N, 8, 3)
    rotated_corners = np.matmul(temp_corners, R_list)  # (N, 8, 3)
    x_corners, y_corners, z_corners = rotated_corners[:, :, 0], rotated_corners[:, :, 1], rotated_corners[:, :, 2]

    x_loc, y_loc, z_loc = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2]

    x = x_loc.reshape(-1, 1) + x_corners.reshape(-1, 8)
    y = y_loc.reshape(-1, 1) + y_corners.reshape(-1, 8)
    z = z_loc.reshape(-1, 1) + z_corners.reshape(-1, 8)

    corners = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1), z.reshape(-1, 8, 1)), axis=2)

    return corners.astype(np.float32)
    
    

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




lidar_2_cam = np.eye(4).astype(np.float32)
lidar_2_cam [:3, :3] = np.array([
    [ -0.488561,   0.872221,  -0.023225],
    [0.056886,   0.005281,  -0.998367],
    [-0.870674,  -0.489084,  -0.052197]
])
lidar_2_cam[:3, 3] = np.array([0.015570,   1.497755,  -0.023615])



# 加载对应的检测结果
pkl_path = '/data/project/OpenPCDet_ziyu/output/15b_models/centerpoint_ped_15b/use3000/eval/eval_with_train/epoch_79/val/result.pkl'
with open(pkl_path, 'rb') as f:
    infos = pickle.load(f)  # 加载pkl数据
info = infos[0]
# import pdb; pdb.set_trace()
boxes_3d = info['boxes_lidar']
# scores = info['score']

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
# # (x,-y,-z+1.5) zhangyuanyi
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
# 这里和图像分辨率有关系
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
