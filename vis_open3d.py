import open3d as o3d
import json
import numpy as np
from scipy.spatial.transform import Rotation as R

def load_pcd_file(file_path):
    """加载PCD格式点云文件"""
    pcd = o3d.io.read_point_cloud(file_path)
    print(f"加载点云文件: {file_path}，包含 {len(pcd.points)} 个点")
    return pcd

def load_labels(file_path):
    """加载标签数据，匹配指定的JSON格式"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 按照提供的格式，标签路径为shapes -> shapes
    return data.get('shapes', {})
    # return data.get('shapes', {}).get('shapes', [])

def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = o3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # 补充线条以完善边界框显示
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    return line_set, box3d

def create_orientation_arrow(box_center, rotation_matrix, length=0.5):
    """创建表示边界框朝向的箭头"""
    # 计算箭头方向
    direction = rotation_matrix[:, 0]  # 使用X轴方向
    direction = direction / np.linalg.norm(direction)
    
    # 箭头起点和终点
    start = box_center
    end = start + direction * length
    
    # 创建箭头
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.03,
        cone_radius=0.07,
        cylinder_height=length*0.8,
        cone_height=length*0.2
    )
    
    # 旋转箭头以匹配边界框朝向
    default_dir = np.array([1, 0, 0])
    if np.dot(default_dir, direction) < 0.999:
        axis = np.cross(default_dir, direction)
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.dot(default_dir, direction))
        arrow.rotate(R.from_rotvec(axis * angle).as_matrix())
    
    arrow.translate(start)
    arrow.paint_uniform_color([1.0, 1.0, 0.0])  # 黄色箭头
    return arrow

def process_label(label):
    """处理单个标签，转换为Open3D可视化对象"""
    # 提取边界框中心
    center = np.array(label['3dpoints'])
    
    # 提取边界框尺寸 (长度、宽度、高度)
    length = label['3dBoxL']
    width = label['3dBoxW']
    height = label['3dBoxH']
    
    # 从四元数获取旋转角度 (z轴旋转)
    angle = label['angle']
    quaternion = [angle['x'], angle['y'], angle['z'], angle['w']]
    rotation = R.from_quat(quaternion)
    euler = rotation.as_euler('xyz', degrees=False)
    z_rotation = euler[2]  # 使用z轴旋转角
    
    # 构建gt_boxes数组 [center_x, center_y, center_z, length, width, height, z_rotation]
    gt_boxes = np.array([
        center[0], center[1], center[2],
        length, width, height,
        z_rotation
    ])
    
    # 转换为Open3D实例
    line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes)
    
    # 设置边界框颜色
    color = label['labelData']['color']
    rgb_color = [
        int(color[1:3], 16)/255.0,
        int(color[3:5], 16)/255.0,
        int(color[5:7], 16)/255.0
    ]
    line_set.paint_uniform_color(rgb_color)
    
    # 创建朝向箭头
    arrow = create_orientation_arrow(center, box3d.R, length=length*0.6)
    
    return line_set, arrow, label['labelData']['aliase']

def visualize_point_cloud_with_labels(pcd_file, labels_file):
    """可视化点云和标签"""
    # 加载数据
    # align_transform = np.array([
    # [-0.1135988,   0.99345154,  0.01222071, -0.04967225],
    # [ 0.9934563,   0.11372859, -0.01050686,  0.23528954],
    # [-0.0118279,   0.01094717, -0.99987012,  0.03008906],
    # [ 0.,          0.,          0.,          1.        ]
    # ])

    # align_transform = np.array([
    #     [-0.98246614,  0.0940865,  -0.16095967,  1.19604319],
    #     [ 0.12184263,  0.97747356, -0.17233633, -0.58192143],
    #     [ 0.1411193,  -0.18892635, -0.97179842,  0.25561647],
    #     [ 0.,          0.,          0.,          1.        ]
    #     ])
    
    # align_transform = np.array(
    #     [[-0.32122805, -0.94515997,  0.05903535,  0.04526527],
    #     [-0.9459984,   0.3231326,   0.02592981,  0.06620629],
    #     [-0.04358407, -0.04751797, -0.99791907, -0.08589521],
    #     [ 0.,          0.,          0.,          1.        ]]
    # )
    pcd = load_pcd_file(pcd_file)
    # pcd.paint_uniform_color([0.5, 0.5, 0.5])
    # pcd=pcd.transform(align_transform)  # 应用变换矩阵


    labels = load_labels(labels_file)
    
    # 准备可视化对象
    vis_objects = [pcd]
    labels_info = []
    
    # 处理每个标签
    for label in labels:
        if label['type'] == 'cube':  # 只处理立方体类型
            try:
                line_set, arrow, label_name = process_label(label)
                vis_objects.append(line_set)
                # vis_objects.append(arrow)
                labels_info.append(f"{label_name} (标签: {label['label']})")
            except Exception as e:
                print(f"处理标签 {label.get('id')} 时出错: {str(e)}")
                continue
    
    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="点云和标签可视化", width=1280, height=720)
    
    # 添加所有对象
    for obj in vis_objects:
        vis.add_geometry(obj)
    
    # 设置可视化参数
    opt = vis.get_render_option()
    # opt.background_color = [0.05, 0.05, 0.05]  # 深灰色背景
    opt.point_size = 2
    opt.show_coordinate_frame = True
    
    # 显示标签信息
    print("\n检测到的标签:")
    for i, info in enumerate(labels_info):
        print(f"{i+1}. {info}")
    
    print("\n操作提示:")
    print("  - 鼠标拖动: 旋转视角")
    print("  - 滚轮: 缩放")
    print("  - Shift+鼠标拖动: 平移")
    print("  - Q: 退出")
    
    # 运行可视化
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='使用自定义边界框函数可视化点云和标签')
    parser.add_argument('pcd_file', help='PCD点云文件路径')
    parser.add_argument('labels_file', help='标签JSON文件路径')
    args = parser.parse_args()
    
    visualize_point_cloud_with_labels(args.pcd_file, args.labels_file)
    