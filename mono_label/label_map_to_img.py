import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import os
from PIL import Image
from matplotlib.patches import Rectangle
import cv2

cn_en_dict = {
    "汽车": "car",
    "椅子": "chair",
    "桌子": "table",
    "自定义": "robot"
}

# --------------------------
# 1. 配置：类别颜色映射和名称映射（核心新增）
# --------------------------
# 类别ID -> BGR颜色（确保颜色区分明显，OpenCV用BGR格式）
class_color_map = {
    "car": (0, 0, 255),    # 类别0：红色（如car）
    "chair": (0, 255, 0),    # 类别1：绿色（如pedestrian）
    "table": (255, 0, 0),    # 类别2：蓝色（如cyclist）
    "robot": (0, 255, 255),  # 类别3：黄色（如truck）
    "trash_can": (255, 0, 255), # 类别4：紫色（如bus）
    "screen": (255, 255, 0) # 类别5：黄色（如screen）
    # 可根据实际类别扩展...
}


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

# 2. 解析相机外参文件
def parse_camera_extrinsics(txt_path):
    """解析相机外参文件"""
    cameras = {}
    with open(txt_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    # 每两行一组：第一行是相机参数，第二行是点信息
    for i in range(0, len(lines), 2):
        if i + 1 >= len(lines):
            break
            
        cam_line = lines[i].split()
        # points_line = lines[i+1].split()
        
        if len(cam_line) < 10:
            continue
            
        image_id = int(cam_line[0])
        # 四元数 (qw, qx, qy, qz)
        quaternion = [float(cam_line[1]), float(cam_line[2]), float(cam_line[3]), float(cam_line[4])]
        # 平移向量 (tx, ty, tz)
        translation = [float(cam_line[5]), float(cam_line[6]), float(cam_line[7])]
        camera_id = int(cam_line[8])
        image_name = cam_line[9]
        
        cameras[image_id] = {
            'image_name': image_name,
            'camera_id': camera_id,
            'quaternion': quaternion,
            'translation': translation
        }
    
    return cameras

# 3. 定义相机内参 (手动设置)
def get_camera_intrinsics():
    """手动设置相机内参"""
    # 示例内参，实际应用中需要根据相机校准数据修改
    # fx = 644.695000  # 焦距x
    # fy = 644.734000  # 焦距y
    # cx = 645.353000   # 主点x坐标
    # cy = 368.456000   # 主点y坐标
    # width = 1280 # 图像宽度
    # height = 720 # 图像高度
    # fx = 605.231000
    # fy = 605.133900  # 焦距y
    # cx = 320.931700   # 主点x坐标
    # cy = 253.091400   # 主点y坐标
    # width = 640 # 图像宽度
    # height = 480 # 图像高度
    width = 512
    height = 384
    fx = 491.61243079/2
    fy = 491.59793421/2
    cx = 509.19576525/2
    cy = 390.55245804/2



    # 内参矩阵
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    return K, width, height

# 4. 坐标转换和投影函数
def quaternion_to_rotation_matrix(q):
    """将四元数转换为旋转矩阵"""
    w, x, y, z = q
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])

def euler_to_rotation_matrix(euler):
    """将欧拉角转换为旋转矩阵 (XYZ顺序)"""
    rx, ry, rz = euler
    
    # 绕X轴旋转
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    
    # 绕Y轴旋转
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    
    # 绕Z轴旋转
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])
    
    # 组合旋转矩阵
    R = Rz @ Ry @ Rx
    return R

def compute_bounding_box(points_2d):
    """计算2D点集的最小外接矩形"""
    min_x = np.min(points_2d[:, 0])
    max_x = np.max(points_2d[:, 0])
    min_y = np.min(points_2d[:, 1])
    max_y = np.max(points_2d[:, 1])
    
    # 计算中心点、宽度和高度
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    width = max_x - min_x
    height = max_y - min_y
    
    return {
        'min_x': min_x, 'max_x': max_x,
        'min_y': min_y, 'max_y': max_y,
        'center_x': center_x, 'center_y': center_y,
        'width': width, 'height': height
    }


def get_cube_vertices(center, dimensions, rotation):
    """计算3D立方体的8个顶点坐标"""
    l, w, h = dimensions
    # 局部坐标系下的8个顶点
    vertices_local = np.array([
        [l/2, w/2, h/2],   # 前右上
        [l/2, w/2, -h/2],  # 后右上
        [l/2, -w/2, h/2],  # 前右下
        [l/2, -w/2, -h/2], # 后右下
        [-l/2, w/2, h/2],  # 前左上
        [-l/2, w/2, -h/2], # 后左上
        [-l/2, -w/2, h/2], # 前左下
        [-l/2, -w/2, -h/2] # 后左下
    ])
    
    # 应用旋转
    R = euler_to_rotation_matrix(rotation)
    vertices_rotated = np.dot(vertices_local, R.T)
    
    # 应用平移（移动到中心点）
    vertices_world = vertices_rotated + np.array(center)
    
    return vertices_world

def project_3d_to_2d(points_3d, K, R, t):
    """将3D点投影到2D图像平面"""
    # 转换为齐次坐标
    points_3d_hom = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    
    # 相机外参矩阵 [R | t]
    extrinsic = np.hstack([R, np.array(t).reshape(3, 1)])

    if False:
        print("extrinsic:", extrinsic)
        extrinsic = np.vstack([extrinsic, [0, 0, 0, 1]])  # 转换为4x4矩阵
        # print("points_3d_hom.T:", points_3d_hom.T)
        extrinsic = np.linalg.inv(extrinsic)  # 转换为从世界坐标到相机坐标的变换矩阵
        extrinsic = extrinsic[:3, :]  # 取前3行
        print("extrinsic.inv:", extrinsic)

    
    # 投影：世界坐标 -> 相机坐标 -> 图像坐标
    points_cam = extrinsic @ points_3d_hom.T
    points_img_hom = K @ points_cam
    
    # 转换为非齐次坐标
    points_img = points_img_hom[:2, :] / points_img_hom[2, :]
    
    return points_img.T

# 5. 可视化函数
def visualize_projection_on_image(cubes, cameras, K, image_folder, image_id=1):
    """在实际图像上可视化3D立方体的投影"""
    if image_id not in cameras:
        print(f"找不到ID为{image_id}的相机参数")
        return None
    
    camera = cameras[image_id]
    image_name = camera['image_name']
    image_path = os.path.join(image_folder, image_name)
    
    # 检查图像文件是否存在
    if not os.path.exists(image_path):
        print(f"警告：图像文件 {image_path} 不存在，将使用空白背景")
        # 使用相机内参中的尺寸创建空白图像
        _, width, height = get_camera_intrinsics()
        img = np.ones((height, width, 3), dtype=np.uint8) * 255  # 白色背景
    else:
        # 加载图像
        img = np.array(Image.open(image_path))
        height, width = img.shape[:2]
    
    print(f"处理图像: {image_name} (尺寸: {width}x{height})")
    
    # 获取相机外参
    R = quaternion_to_rotation_matrix(camera['quaternion'])
    t = camera['translation']
    
    # 立方体的边连接关系
    edges = [
        [0, 1], [1, 3], [3, 2], [2, 0],  # 前面和后面
        [4, 5], [5, 7], [7, 6], [6, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]   # 连接前后
    ]
    
    # 存储当前图像的所有YOLO标签
    yolo_labels = []

    # 处理每个立方体
    for cube in cubes:
        # print(cube)
        # 计算立方体顶点
        vertices = get_cube_vertices(
            cube['center'], 
            cube['dimensions'], 
            cube['rotation']
        )
        
        # 投影到2D图像
        vertices_2d = project_3d_to_2d(vertices, K, R, t)
        
        # 过滤掉在相机后方的点 (z < 0)
        points_cam = R @ vertices.T + np.array(t).reshape(3, 1)
        visible = points_cam[2, :] > 0

        if not np.any(visible):
            continue  # 完全不可见的立方体不处理

        # 获取可见的2D顶点
        visible_vertices_2d = vertices_2d[visible]
        bbox = compute_bounding_box(visible_vertices_2d)


        ## 绘制标签方式改这里,立方体为True  | 矩形框为False
        if True:
            # 绘制立方体投影边
            for edge in edges:
                v1, v2 = edge
                if visible[v1] and visible[v2]:
                    # 确保点在图像范围内
                    if (0 <= vertices_2d[v1, 0] <= width and 0 <= vertices_2d[v1, 1] <= height and
                        0 <= vertices_2d[v2, 0] <= width and 0 <= vertices_2d[v2, 1] <= height):
                        ax.plot(
                            [vertices_2d[v1, 0], vertices_2d[v2, 0]],
                            [vertices_2d[v1, 1], vertices_2d[v2, 1]],
                            color=cube['color'],
                            linewidth=2,
                            alpha=0.8
                        )
        
        else:

            min_x_raw = bbox['min_x']
            max_x_raw = bbox['min_x'] + bbox['width']  # 原始右边界
            min_y_raw = bbox['min_y']
            max_y_raw = bbox['min_y'] + bbox['height']  # 原始下边界
            
            # 2. 裁剪到图像范围内（核心步骤）
            min_x_clamped = np.clip(min_x_raw, 0, width - 1)  # 左边界不小于0，不大于width-1
            max_x_clamped = np.clip(max_x_raw, 0, width - 1)  # 右边界同理
            min_y_clamped = np.clip(min_y_raw, 0, height - 1)  # 上边界不小于0，不大于height-1
            max_y_clamped = np.clip(max_y_raw, 0, height - 1)  # 下边界同理
            
            # 3. 计算裁剪后的宽高，过滤无效框（宽高≤0则完全在图像外）
            clamped_width = max_x_clamped - min_x_clamped
            clamped_height = max_y_clamped - min_y_clamped
            if clamped_width <= 0 or clamped_height <= 0:
                continue  # 完全在图像外，跳过
            
            # 4. 更新边界框为裁剪后的值
            bbox_clamped = {
                'min_x': min_x_clamped,
                'min_y': min_y_clamped,
                'width': clamped_width,
                'height': clamped_height
            }
            
            # --------------------------
            # 绘制裁剪后的矩形框（OpenCV）
            # --------------------------
            pt1 = (int(bbox_clamped['min_x']), int(bbox_clamped['min_y']))  # 左上角（裁剪后）
            pt2 = (int(max_x_clamped), int(max_y_clamped))  # 右下角（裁剪后）
            cv2.rectangle(img, pt1, pt2, class_color_map[cube['label']], 2)
            
            # 文本位置：矩形框左上角上方10px（避免重叠），若超出顶部则放在框内
            text_pos = (pt1[0], pt1[1] - 10) if pt1[1] - 10 > 0 else (pt1[0], pt1[1] + 20)
            
            # 文本样式：字体、大小、颜色（白色文本+黑色边框，更清晰）
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            text_color = (255, 255, 255)  # 白色文本
            thickness = 1
            # 先绘制黑色描边（增加可读性）
            # cv2.putText(img, class_name, text_pos, font, font_scale, (0, 0, 0), thickness + 1)
            # 再绘制白色文本
            cv2.putText(img, cube['label'], text_pos, font, font_scale, text_color, thickness)

            # --------------------------
            # 生成裁剪后的YOLO标签
            # --------------------------
            class_id = cube['label']  # 类别ID（需确保为整数）
            
            # 计算裁剪后的中心坐标（归一化前）
            x_center_pixel = min_x_clamped + clamped_width / 2.0
            y_center_pixel = min_y_clamped + clamped_height / 2.0
            
            # 归一化（确保在0~1范围内）
            x_center = x_center_pixel / width
            y_center = y_center_pixel / height
            w_norm = clamped_width / width
            h_norm = clamped_height / height
            
            # 再次裁剪归一化值（理论上已在0~1，保险措施）
            x_center = np.clip(x_center, 0.0, 1.0)
            y_center = np.clip(y_center, 0.0, 1.0)
            w_norm = np.clip(w_norm, 0.0, 1.0)
            h_norm = np.clip(h_norm, 0.0, 1.0)
            
            # 添加到YOLO标签
            yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
            yolo_labels.append(yolo_line)
        


        # # 添加标签
        # if np.any(visible):
        #     # 找到可见顶点的中心作为标签位置
        #     visible_vertices = vertices_2d[visible]
        #     label_pos = np.mean(visible_vertices, axis=0)
            
        #     # 确保标签在图像范围内
        #     if 0 <= label_pos[0] <= width and 0 <= label_pos[1] <= height:
        #         ax.text(
        #             label_pos[0], label_pos[1], 
        #             f"{cube['label']}",
        #             # f"{cn_en_dict[cube['label_name']]}",
        #             color=cube['color'],
        #             fontweight='bold',
        #             bbox=dict(facecolor='white', alpha=0.7, pad=2)
        #         )
    

    label_dir = image_folder.replace("images","2D_label")  # 标签文件夹路径
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    label_path = os.path.join(label_dir, f"{image_name}.txt")  # 标签文件路径

    # 写入标签（若没有目标则创建空文件）
    with open(label_path, 'w', encoding='utf-8') as f:
        if yolo_labels:
            f.write('\n'.join(yolo_labels) + '\n')  # 每行一个目标
        else:
            f.write('')  # 无目标则为空文件


    # ## 显示图像（可选）
    # cv2.imshow("2D Bounding Boxes", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 保存绘制后的图像（可选）
    # cv2.imwrite(os.path.join(img_dir, f"{image_name}_labeled.jpg"), img)


    # plt.tight_layout()
    return img

# 6. 主函数
def main(json_path, camera_txt_path, image_folder):
    # 解析数据
    cubes = parse_3d_annotations(json_path)
    cameras = parse_camera_extrinsics(camera_txt_path)
    K, _, _ = get_camera_intrinsics()
    
    print(f"解析到 {len(cubes)} 个3D立方体")
    print(f"解析到 {len(cameras)} 个相机参数")
    print(f"图像文件夹: {image_folder}")
    
    # 为每个相机可视化投影结果
    for image_id in cameras:
        fig = visualize_projection_on_image(cubes, cameras, K, image_folder, image_id)
        if True:
            # if image_id >= 71:
            #     plt.show()
            
            # 可选：保存带投影的图像
            output_name = f"projected_{cameras[image_id]['image_name']}"
            output_path = os.path.join(image_folder.replace("images","traj_project_images_3D"),output_name)
            os.makedirs(image_folder.replace("images","traj_project_images_3D"), exist_ok=True)
            
            # fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
            cv2.imwrite(output_path, fig)
            print(f"带投影的图像已保存到 {output_path}")





if __name__ == "__main__":
    # 手动设置文件路径
    json_annotation_path = "/mnt/dln/data/datasets/0915/for_bev_test/bev_label.json"  # 3D点云标注JSON文件路径
    camera_extrinsics_path = "/mnt/dln/data/datasets/0915/for_bev_test/Colmap/0/sparse/0/images.txt"  # 相机外参文件路径
    image_folder_path = "/mnt/dln/data/datasets/0915/for_bev_test/fixed_images/"  # 图像文件夹路径
    
    # json_annotation_path = "/mnt/dln/data/datasets/0915/make_label_raw/label/meeting_414-33244.json"
    # camera_extrinsics_path = "/mnt/dln/data/datasets/0915/make_label_raw/meeting_414/sparse/0/images.txt"
    # image_folder_path = "/mnt/dln/data/datasets/0915/make_label_raw/meeting_414/images/"

    # 检查文件是否存在
    main(json_annotation_path, camera_extrinsics_path, image_folder_path)
    
    # 改262行，True为立方体，False为矩形框