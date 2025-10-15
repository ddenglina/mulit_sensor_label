import open3d as o3d


pcd = o3d.io.read_point_cloud("/mnt/dln/ros2_ws/src/FAST-LIVO2/Log/PCD/all_raw_points.pcd")

# pcd.paint_uniform_color([0, 0, 0])

# pcd.estimate_normals()
# FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
o3d.visualization.draw_geometries([pcd])
o3d.io.write_point_cloud("/mnt/dln/ros2_ws/src/FAST-LIVO2/Log/PCD/all_raw_points.ply", pcd)
# o3d.visualization.draw_geometries([pcd])  # 显示坐标系

