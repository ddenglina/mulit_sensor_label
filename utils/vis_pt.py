import torch
import open3d as o3d

# 加载SLAM地图数据
slam_map = torch.load("/mnt/dln/projects/perception_fusion/data/vipe/metal_earphone/vipe/metal_earphone_slam_map.pt")
# 查看数据结构（通常是字典类型）
print("地图数据包含的键值:", slam_map.keys())

# dict_keys(['dense_disp_xyz', 'dense_disp_rgb', 'dense_disp_packinfo', 'dense_disp_frame_inds', 'device'])
xyz_map = slam_map["dense_disp_xyz"]
rgb_map = slam_map["dense_disp_rgb"]
print("xyz_map:",xyz_map.shape)
print("rgb_map:",rgb_map.shape)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz_map)
pcd.colors = o3d.utility.Vector3dVector(rgb_map)

o3d.visualization.draw_geometries([pcd])
o3d.io.write_point_cloud("/mnt/dln/projects/perception_fusion/data/vipe/metal_earphone/vipe/metal_earphone_slam_map.ply", pcd)


