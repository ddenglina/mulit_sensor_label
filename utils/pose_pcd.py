
import os
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import json
import math


def get_pose(pose_path):
    with open(pose_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            pose = line.strip().split()
            pose = pose[1:]
            quaternion = [float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6])]  # 四元数 (qx, qy, qz, qw)
            translation = [float(pose[0]), float(pose[1]), float(pose[2])]  # 平移向量 (tx, ty, tz)

        # 外参矩阵
        rotation_matrix = R.from_quat(quaternion).as_matrix()
        extrinsics_matrix = np.eye(4)
        extrinsics_matrix[:3, :3] = rotation_matrix
        extrinsics_matrix[:3, 3] = translation

        pose={'quaternion':quaternion,'translation':translation,'extrinsics_matrix':extrinsics_matrix}    
    return pose


pose_path = "/mnt/dln/ros2_ws/src/FAST-LIVO2/Log/label/pose/1.txt"
pose = get_pose(pose_path)
pcd = o3d.io.read_point_cloud("/mnt/dln/ros2_ws/src/FAST-LIVO2/Log/label/pcd/1.pcd")
pcd.paint_uniform_color([1, 0, 0])
pcd=pcd.transform(pose['extrinsics_matrix'])



pose_path_100 = "/mnt/dln/ros2_ws/src/FAST-LIVO2/Log/label/pose/100.txt"
pose_100 = get_pose(pose_path_100)
# pcd.transform(np.linalg.inv(pose['extrinsics_matrix']))
pcd1 = o3d.io.read_point_cloud("/mnt/dln/ros2_ws/src/FAST-LIVO2/Log/label/pcd/100.pcd")
pcd1=pcd1.transform(pose_100['extrinsics_matrix'])

pcd2 = o3d.io.read_point_cloud("/mnt/dln/ros2_ws/src/FAST-LIVO2/Log/PCD/all_raw_points.pcd")
# pcd2=pcd2.transform(pose_200['extrinsics_matrix'])
o3d.visualization.draw_geometries([pcd, pcd1, pcd2])


# 这种是正确的，可以合并在一起

