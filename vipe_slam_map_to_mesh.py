import numpy as np
import open3d as o3d
import os


# 点云去噪函数 - 保留颜色信息
def denoise_point_cloud(pcd, method='statistical', **kwargs):
    """
    对点云进行去噪处理，同时保留颜色信息
    
    参数:
        pcd: Open3D点云对象
        method: 去噪方法，可选'statistical'（统计滤波）或'radius'（半径滤波）
        kwargs: 对应方法的参数
    返回:
        去噪后的点云对象（包含颜色信息）
    """
    # 检查原始点云是否有颜色
    has_colors = len(pcd.colors) > 0 and len(pcd.colors) == len(pcd.points)
    
    if method == 'statistical':
        # 统计滤波参数
        nb_neighbors = kwargs.get('nb_neighbors', 20)
        std_ratio = kwargs.get('std_ratio', 2.0)
        # 获取内点索引而非直接获取滤波后的点云
        _, ind = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, 
            std_ratio=std_ratio
        )
        
    elif method == 'radius':
        # 半径滤波参数
        nb_points = kwargs.get('nb_points', 16)
        radius = kwargs.get('radius', 0.05)
        # 获取内点索引而非直接获取滤波后的点云
        _, ind = pcd.remove_radius_outlier(
            nb_points=nb_points, 
            radius=radius
        )
        
    else:
        raise ValueError("不支持的去噪方法，可选方法：'statistical' 或 'radius'")
    
    # 根据索引创建新的点云并保留颜色
    denoised_pcd = o3d.geometry.PointCloud()
    # 保留点坐标
    denoised_pcd.points = o3d.utility.Vector3dVector(
        np.asarray(pcd.points)[ind]
    )
    
    # 如果原始点云有颜色，保留颜色信息
    if has_colors:
        denoised_pcd.colors = o3d.utility.Vector3dVector(
            np.asarray(pcd.colors)[ind]
        )
        
    print(f"滤波后保留 {len(denoised_pcd.points)} 个点")
    return denoised_pcd


if __name__ == "__main__":
    # 读取点云文件
    test_file = "/mnt/dln/projects/perception_fusion/data/vipe/metal_earphone/vipe/metal_earphone.ply"
    pcd = o3d.io.read_point_cloud(test_file)
    print(f"原始点云包含 {len(pcd.points)} 个点")
    
    # 检查是否有颜色信息
    if len(pcd.colors) > 0:
        print("原始点云包含颜色信息")
    else:
        print("原始点云不包含颜色信息")

    # 执行去噪
    denoised_pcd = denoise_point_cloud(
        pcd, 
        method='statistical', 
        nb_neighbors=20,  # 可根据点云密度调整
        std_ratio=0.6     # 阈值越小，去噪越激进
    )
    merged_pcd = denoised_pcd.voxel_down_sample(voxel_size=0.0001)  # 0.01:6365, 0.005:30382, 0.002:285166
    print(f"down_sample_pcd: {len(merged_pcd.points)} ")
    merged_pcd.estimate_normals()
    merged_pcd.orient_normals_consistent_tangent_plane(100)   # 法向量方向一致化(必须)    denoised_pcd
    o3d.visualization.draw_geometries([merged_pcd], window_name="pcd_down",point_show_normal=True)

    radii = [0.005, 0.01, 0.02, 0.04]
    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
               merged_pcd, o3d.utility.DoubleVector(radii))
    o3d.visualization.draw_geometries([rec_mesh])

    # # 可视化结果
    # o3d.visualization.draw_geometries([denoised_pcd], window_name="保留颜色的去噪点云")
    
    # # 保存去噪后的点云（包含颜色）
    # output_file = test_file.replace(".ply", "_denoised_with_color.ply")
    # o3d.io.write_point_cloud(output_file, denoised_pcd)
    # print(f"去噪后的点云已保存至: {output_file}")


    # 保存网格为PLY文件
    mesh_path = os.path.join("/mnt/dln/projects/perception_fusion/data/vipe/metal_earphone/vipe/mesh_metal_earphone.ply")
    o3d.io.write_triangle_mesh(mesh_path, rec_mesh)