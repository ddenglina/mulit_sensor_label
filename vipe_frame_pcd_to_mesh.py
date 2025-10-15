import os
import open3d as o3d    



lidar_path = "/mnt/dln/results/week34_20250901/vipe/result/frame_0006_crop.ply"

pcd = o3d.io.read_point_cloud(lidar_path)

merged_pcd = pcd.voxel_down_sample(voxel_size=0.0005)  # 0.01:6365, 0.005:30382, 0.002:285166
print(f"down_sample_pcd: {len(merged_pcd.points)} ")
merged_pcd.estimate_normals()
merged_pcd.orient_normals_consistent_tangent_plane(100)   # 法向量方向一致化(必须)    denoised_pcd
o3d.visualization.draw_geometries([merged_pcd], window_name="pcd_down",point_show_normal=True)

radii = [0.005, 0.01, 0.02, 0.04]
rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            merged_pcd, o3d.utility.DoubleVector(radii))
o3d.visualization.draw_geometries([rec_mesh])

# 保存网格为PLY文件
mesh_path = os.path.join("/mnt/dln/results/week34_20250901/vipe/result//mesh_frame_00006.ply")
o3d.io.write_triangle_mesh(mesh_path, rec_mesh)