import os
import open3d as o3d
import numpy as np


lidar_dir = "/mnt/dln/projects/perception_fusion/data/vipe/pcd/point_cloud_frames_metal_earphone/"
pc_file = os.path.join(lidar_dir, "frame_0000.ply")
pc_file_2 = os.path.join(lidar_dir, "frame_0300.ply")

pcd = o3d.io.read_point_cloud(pc_file)
pcd_2 = o3d.io.read_point_cloud(pc_file_2)


# trans =  np.array([[ 1.0000000e+00, -1.0869569e-05,  5.7116074e-05, -4.0429888e-05],
#             [ 1.0870857e-05,  1.0000000e+00, -2.2552982e-05,  3.1800239e-05],
#             [-5.7115827e-05,  2.2553604e-05,  1.0000000e+00,  2.9197172e-05],
#             [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]])
trans_2 = np.array([[-0.9270332,   0.2623166,  -0.26795423,  0.6304798 ],
                [-0.2282477,   0.17221296,  0.95825136, -1.5500367 ],
                [ 0.29751045,  0.9494908,  -0.099774,    1.2244457 ],
                [ 0.,          0. ,         0.,          1.        ]])

# pcd = pcd.transform(np.linalg.inv(trans))
pcd_2 = pcd_2.transform(np.linalg.inv(trans_2))
# transformation = np.array([[ 9.16400552e-01, -3.12589407e-02,  3.99040043e-01,  1.66251417e-03],
#                                 [ 7.18979985e-02,  9.93585706e-01, -8.72818008e-02, -2.33045053e-02],
#                                 [-3.93752158e-01,  1.08675264e-01,  9.12769914e-01,  4.66303958e-04],
#                                 [ 0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  1.00000000e+00]])    
# transformation = np.linalg.inv(transformation)
# pcd=pcd.transform(transformation)
    

# [[ 9.1639125e-01 -3.1262487e-02  3.9906111e-01  1.6801446e-03]
#  [ 7.1908496e-02  9.9358416e-01 -8.7290913e-02 -2.3316531e-02]
#  [-3.9377186e-01  1.0868850e-01  9.1275984e-01  5.6757382e-04]
#  [ 0.0000000e+00  0.0000000e+00  0.0000000e+00  1.0000000e+00]]

o3d.visualization.draw_geometries([pcd, pcd_2])  # 显示坐标系
# o3d.io.write_point_cloud("output.ply", [pcd, pcd_2])