import os
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import json
import math


# 辅助函数：将3D点通过变换矩阵进行变换
def transform_point(point, matrix):
    """
    对3D点应用4x4变换矩阵
    point: 三维坐标 [x, y, z]
    matrix: 4x4变换矩阵
    return: 变换后的三维坐标 [x', y', z']
    """
    # 转换为齐次坐标
    homogeneous_point = np.array([point[0], point[1], point[2], 1.0])
    # 应用变换矩阵
    transformed_homogeneous = matrix @ homogeneous_point
    # 转换回3D坐标
    return transformed_homogeneous[:3]

# 假设parse_3d_annotations和create_bounding_box函数已定义


# 读标签文件
# 1. 解析3D点云标注的JSON文件
def parse_3d_annotations(json_path):
    """解析3D立方体标注的JSON文件"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    cubes = []
    for shape in data['shapes']:
        if shape['type'] == 'cube':
            cube = {
                'id': shape['id'],
                'label': shape['label'],
                'label_name': shape['labelData']['aliase'],
                'color': shape['labelData']['color'],
                'center': shape['3dpoints'],  # [x, y, z]
                'dimensions': [shape['3dBoxL'], shape['3dBoxW'], shape['3dBoxH']],  # [长度, 宽度, 高度]
                'rotation': [shape['rotationX'], shape['rotationY'], shape['rotationZ']],  # [x, y, z]旋转角度(弧度)
                'quaternion': [shape['angle']['w'], shape['angle']['x'], shape['angle']['y'], shape['angle']['z']]  # [w, x, y, z]
            }
            cubes.append(cube)
    
    return cubes


def create_bounding_box(center, dimensions, rotation):
    """
    创建3D边界框
    
    参数:
        center: 边界框中心点坐标 [x, y, z]
        dimensions: 边界框尺寸 [长度, 宽度, 高度]
        rotation: 边界框旋转角度 [x, y, z] (弧度)
    
    返回:
        open3d.geometry.OrientedBoundingBox对象
    """
    # 创建轴对齐边界框
    bbox = o3d.geometry.OrientedBoundingBox(
        center=np.array(center),
        R=o3d.geometry.get_rotation_matrix_from_xyz(rotation),
        extent=np.array(dimensions),
    )
    
    return bbox


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
    
def hex_to_rgb(hex_value):
    # 去除十六进制颜色值中的'#'符号
    hex_value = hex_value.lstrip('#')

    # 从十六进制字符串中提取红、绿、蓝通道的值
    red = int(hex_value[0:2], 16)
    green = int(hex_value[2:4], 16)
    blue = int(hex_value[4:6], 16)

    # 返回RGB值
    return red, green, blue


def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 
 
def rot2euler(R):
    assert (isRotationMatrix(R))
 
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
 
    singular = sy < 1e-6
 
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2]) * 180 / np.pi
        y = math.atan2(-R[2, 0], sy) * 180 / np.pi
        z = math.atan2(R[1, 0], R[0, 0]) * 180 / np.pi
    else:
        x = math.atan2(-R[1, 2], R[1, 1]) * 180 / np.pi
        y = math.atan2(-R[2, 0], sy) * 180 / np.pi
        z = 0
    return np.array([x, y, z])

def create_bbox_axes(bbox, axis_length=1.0,color=[1, 0, 0]):
    """
    为边界框创建朝向指示坐标轴（LineSet）
    axis_length: 坐标轴长度（可根据边界框大小调整）
    """
    # 边界框中心（坐标轴起点）
    center = bbox.center
    
    # 从旋转矩阵提取朝向向量（每一列对应x、y、z轴的方向）
    x_axis = bbox.R[:, 0] * axis_length  # x轴方向向量
    y_axis = bbox.R[:, 1] * axis_length  # y轴方向向量
    z_axis = bbox.R[:, 2] * axis_length  # z轴方向向量
    
    # 坐标轴终点
    x_end = center + x_axis
    y_end = center + y_axis
    z_end = center + z_axis
    
    # 创建线段集合（LineSet）
    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector([
        center, x_end,  # x轴线段
        center, y_end,  # y轴线段
        center, z_end   # z轴线段
    ])
    
    # 定义线段索引（0-1: x轴, 2-3: y轴, 4-5: z轴）
    # lines.lines = o3d.utility.Vector2iVector([
    #     [0, 1], [2, 3], [4, 5]
    # ])

    # 只画X轴
    lines.lines = o3d.utility.Vector2iVector([
        [0, 1]
    ])
    lines.colors = o3d.utility.Vector3dVector([
        color
    ])
    # 设置颜色（x轴红, y轴绿, z轴蓝）
    # lines.colors = o3d.utility.Vector3dVector([
    #     [1, 0, 0],  # x轴红色
    #     [0, 1, 0],  # y轴绿色
    #     [0, 0, 1]   # z轴蓝色
    # ])
    
    return lines

def main(json_annotation_path,pcd_dir,pose_dir,save_label_dir,align_transform=None, min_points=32):

    # 解析标签并对标签进行变换
    cubes = parse_3d_annotations(json_annotation_path)

    files = os.listdir(pose_dir)
    # files = sorted(files)
    for filename in files:

        save_label_path = os.path.join(save_label_dir,filename)
        pose_path = os.path.join(pose_dir,filename)
        pcd_path = os.path.join(pcd_dir,filename.replace("txt","pcd"))
        print(pcd_path)

        # 读取点云
        frame_pcd = o3d.io.read_point_cloud(pcd_path)
        # print(type(frame_pcd))
        frame_pcd.paint_uniform_color([0.5, 0.5, 0.5])
         

        # 读取位姿
        pose = get_pose(pose_path)
        vis_objects = []


        with open(save_label_path, 'w', encoding='utf-8') as f:
            # 写入表头
            f.write("label,center_x,center_y,center_z,extent_x,extent_y,extent_z,R_x,R_y,R_z\n")
            for id,cube in enumerate(cubes):                    
                ####################################################################################

                # 提取原始边界框参数
                center = cube["center"]
                dimensions = cube["dimensions"]
                original_rotation = cube["rotation"]
                label_name = cube["label"]

                # 如果建图的第一帧与点云的第一帧不同，需要对标签进行变换align_transform
                if align_transform is not None:
                    # print("center align_transform...")
                    transfrom_center = transform_point(center, np.linalg.inv(align_transform@pose["extrinsics_matrix"]))
                    # transfrom_center = transform_point(center, np.linalg.inv(align_transform@pose["extrinsics_matrix"]))
                    center = transfrom_center
                else:
                    transfrom_center = transform_point(center, np.linalg.inv(pose["extrinsics_matrix"]))
                    center = transfrom_center

                bbox = create_bounding_box(center, dimensions, original_rotation)
                # # # 对边界框进行旋转
                if align_transform is not None:
                    # print("rotate align_transform...")
                    bbox.rotate(np.linalg.inv(align_transform[:3,:3]@pose["extrinsics_matrix"][:3,:3]))
                else:
                    bbox.rotate(np.linalg.inv(pose["extrinsics_matrix"][:3,:3]))

                # ## 增加框中是否有点云判断 #########################################################
                points_translated = np.asarray(frame_pcd.points) - bbox.center
                points_local = np.dot(points_translated, bbox.R)  # 转换到局部坐标系
                half_extent = bbox.extent / 2
                in_bbox = np.all(np.abs(points_local) <= half_extent + 1e-6, axis=1)
                num_in_box = np.sum(in_bbox)
                
                # 如果点数 >= min_points，则保留该框
                if num_in_box < min_points:
                    continue

                ##可视化###########################################################################################
                # 为边界框分配随机颜色
                if False:
                    color = hex_to_rgb(cube["color"])
                    bbox.color = [c/255 for c in color]
                    axis_length = max(cube["dimensions"]) * 0.8
                    axes = create_bbox_axes(bbox, axis_length,bbox.color)
                    vis_objects.append(bbox)
                    vis_objects.append(axes)


                ##保存标签#########################################################################################
                # print(f'bbox.center:{bbox.center},bbox.extent:{bbox.extent},bbox.R:{bbox.R}')
                # 提取边界框信息并格式化
                # 中心点坐标
                cx, cy, cz = bbox.center
                # 范围
                ex, ey, ez = bbox.extent
                # 旋转矩阵（3x3）展平为列表
                # r_flat = [elem for row in bbox.R for elem in row]
                rx,ry,rz = rot2euler(bbox.R)     
                
                # 拼接成行数据（使用逗号分隔，便于后续后读取）
                line = f"{label_name},{cx:.6f},{cy:.6f},{cz:.6f},{ex:.6f},{ey:.6f},{ez:.6f},{rx:.6f},{ry:.6f},{rz:.6f}\n"
                # line += ",".join(map(str, r_flat)) + "\n"
                
                # 写入文件
                f.write(line)

        # if filename in ["1.txt","100.txt"]:
        #     # 可视化结果
        #     o3d.visualization.draw_geometries([frame_pcd]+vis_objects,window_name=filename)




if __name__=="__main__":
    
    # ai_label_dir = "/mnt/dln/data/datasets/0915/make_label_raw/label/"
    # raw_data_dir = "/mnt/dln/data/datasets/0915/make_label_raw/"
    ai_label_dir = "/mnt/dln/data/datasets/0915/for_bev_lidar_test/label/"
    raw_data_dir = "/mnt/dln/data/datasets/0915/"
    json_annotation_list = os.listdir(ai_label_dir)


    for json_annotation in json_annotation_list:
        json_annotation_path = os.path.join(ai_label_dir,json_annotation)
        # scene_name = json_annotation[:-11]
        scene_name = json_annotation[:-5]
        pcd_dir = os.path.join(raw_data_dir,scene_name,"pcd/")
        pose_dir = os.path.join(raw_data_dir,scene_name,"pose/")
        save_label_dir = os.path.join(raw_data_dir,scene_name,"3D_label/")
        if not os.path.exists(save_label_dir):
            os.makedirs(save_label_dir)
        if len(os.listdir(save_label_dir))!=0:
            continue

        # # # 变换矩阵，# 如果建图的第一帧与点云的第一帧不同，需要对标签进行变换align_transform
        # align_transform = np.array(
        #     [[ 0.65059434, -0.75934361, -0.0111485,  -0.52345514],
        #     [ 0.75942039,  0.65057344,  0.00590458,  0.20729472],
        #     [ 0.00276931, -0.01230788,  0.99992042,  0.01308885],
        #     [ 0.        ,  0.        ,  0.        ,  1.        ]]
        # ) 

        # align_transform = np.array([
        # [ 0.65433284,  0.75163899, -0.08299007,  0.10764468],
        # [-0.75597532,  0.65289601, -0.04720301, -0.34554929],
        # [ 0.01870426,  0.09362492,  0.99543183, -0.1275272 ],
        # [ 0. ,         0. ,         0.  ,        1.        ]])
        align_transform = None
        
        # 3D框：默认设置min_points=35
        # for 2D框: 默认设置min_points=60
               
        main(json_annotation_path,pcd_dir,pose_dir,save_label_dir,min_points=35,align_transform=align_transform)
    
