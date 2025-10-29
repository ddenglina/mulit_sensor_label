
import os
import json
import copy
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree


def FPFH_Compute(pcd, voxel_size):
    radius_normal = voxel_size * 2  # kdtree参数，用于估计法线的半径，
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
# 估计法线的1个参数，使用混合型的kdtree，半径内取最多30个邻居
    radius_feature = voxel_size * 5  # kdtree参数，用于估计FPFH特征的半径
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd,
    o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))  # 计算FPFH特征,搜索方法kdtree
    return pcd_fpfh  # 返回FPFH特征
 
 
# ----------------RANSAC配准--------------------
def execute_global_registration(source, target, source_fpfh,
                                target_fpfh, voxel_size):  # 传入两个点云和点云的特征
    distance_threshold = voxel_size * 1.5  # 设定距离阈值
    print("we use a liberal distance threshold %.3f." % distance_threshold)
# 2个点云，两个点云的特征，距离阈值，一个函数，4，
# 一个list[0.9的两个对应点的线段长度阈值，两个点的距离阈值]，
# 一个函数设定最大迭代次数和最大验证次数
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, source_fpfh, target_fpfh, True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 50000))
    return result



def get_align_transformation(pc1_path, pc2_path):
    """
    Get the transformation matrix to align the reconstructed mesh to the ground truth mesh.
    egistration_icp(source, target, max_correspondence_distance, init, estimation_method, criteria)
    """
    # pc2是map
    pcd1 = o3d.io.read_point_cloud(pc1_path)
    pcd2 = o3d.io.read_point_cloud(pc2_path)
    # pcd2.estimate_normals()
    pcd1.paint_uniform_color([1, 0, 0])

    # rotation_matrix = np.array([
    #     [1, 0, 0],  # X轴反向
    #     [0, -1, 0],  # Y轴反向
    #     [0, 0, -1]    # Z轴保持不变
    # ], dtype=np.float32)
    # pcd1.rotate(rotation_matrix, center=(0, 0, 0))

    # theta = np.pi / 2  # 90°转弧度（若需逆时针旋转，保持theta；顺时针则改为 -np.pi/2）
    # mat_rotate_z_90 = np.array([
    #     [np.cos(theta), -np.sin(theta), 0],  # X轴映射：cosθ*X - sinθ*Y
    #     [np.sin(theta),  np.cos(theta), 0],  # Y轴映射：sinθ*X + cosθ*Y
    #     [0,              0,             1]   # Z轴不变
    # ], dtype=np.float32)
    # pcd1.rotate(mat_rotate_z_90, center=(0, 0, 0))


    ds_pc2 = pcd2.voxel_down_sample(0.5)
    o3d.visualization.draw_geometries([pcd1, ds_pc2])
    
    voxel_size = 0.5
    ds_pc1 = pcd1.voxel_down_sample(voxel_size)
    # ds_pc2 = ds_pc2.voxel_down_sample(voxel_size)
    # o3d.visualization.draw_geometries([ds_pc1, ds_pc2])


    pc1_down_fpfh = FPFH_Compute(ds_pc1, voxel_size)
    pc2_down_fpfh = FPFH_Compute(ds_pc2, voxel_size)
    result_coarse = execute_global_registration(ds_pc1, ds_pc2, pc1_down_fpfh, pc2_down_fpfh, voxel_size)
    
    # temp用于可视化
    pc1_temp = copy.deepcopy(ds_pc1)   
    pc2_temp = copy.deepcopy(ds_pc2)
    pc1_temp.transform(result_coarse.transformation)
    # o3d.visualization.draw_geometries([pc1_temp, pc2_temp])


    threshold = 0.1
    reg_point2plane = o3d.pipelines.registration.registration_icp(
        ds_pc1,
        ds_pc2,
        threshold,
        result_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),  # 点对面方法
    )
    print("point2plane icp results:",reg_point2plane)

    reg_p2p = o3d.pipelines.registration.registration_icp(
        ds_pc1,
        ds_pc2,
        threshold,
        reg_point2plane.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),  # 点对点方法
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 2000))
    
    print("icp results:",reg_p2p)
    transformation = reg_p2p.transformation  # 变换矩阵
    o3d.visualization.draw_geometries([ds_pc1.transform(transformation), pc2_temp])
    return transformation


if __name__=="__main__":

    pc1_path = "/mnt/dln/data/datasets/0915/for_bev_test/down_raw_points.pcd"  # 新的地图
    pc2_path = "/mnt/dln/data/datasets/0915/bev_label/velodyne/down_output_for_lable.pcd" # 原始地图
    transformation = get_align_transformation(pc1_path, pc2_path)

    print(transformation)
    print(np.linalg.inv(transformation))

    pcd1 = o3d.io.read_point_cloud(pc1_path)
    pcd1 = pcd1.transform(transformation)
    pcd1.paint_uniform_color([0.5, 0.5, 0.5])
    pcd2 = o3d.io.read_point_cloud(pc2_path)
    # pcd2 = pcd2.transform(transformation)
    o3d.visualization.draw_geometries([pcd1, pcd2])
