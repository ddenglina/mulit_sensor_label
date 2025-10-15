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
    fx = 605.231000
    fy = 605.133900  # 焦距y
    cx = 320.931700   # 主点x坐标
    cy = 253.091400   # 主点y坐标
    width = 640 # 图像宽度
    height = 480 # 图像高度



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

def project_3d_to_2d(points_3d, K, R, t,label_dir):
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
def visualize_projection_on_image(cubes, cameras, K, image_folder, image_id=1, label_dir="2D_label",is_rotated_180=False):
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
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # 转换为BGR格式以便OpenCV使用
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
        vertices_2d = project_3d_to_2d(vertices, K, R, t,label_dir)
        
        # 过滤掉在相机后方的点 (z < 0)
        points_cam = R @ vertices.T + np.array(t).reshape(3, 1)
        visible = points_cam[2, :] > 0

        if not np.any(visible):
            continue  # 完全不可见的立方体不处理

        # 获取可见的2D顶点
        visible_vertices_2d = vertices_2d[visible]
        bbox = compute_bounding_box(visible_vertices_2d)


        ## 绘制标签方式改这里,立方体为True  | 矩形框为False
        if False:
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

            min_x_original = bbox['min_x']
            min_y_original = bbox['min_y']
            clamped_width = bbox['width']  # 原始宽（未旋转图像中）
            clamped_height = bbox['height']  # 原始高（未旋转图像中）
            max_x_original = min_x_original + clamped_width
            max_y_original = min_y_original + clamped_height
            
            # 2. 计算原始框面积，防护面积为0的异常情况
            original_area = clamped_width * clamped_height
            if original_area <= 0:
                continue  # 原始框无效，跳过

            # 2. 转换坐标到180度旋转后的图像
            if is_rotated_180:
                # 旋转后坐标（x和y轴均反转）
                min_x_rot = width - 1 - min_x_original
                max_x_rot = width - 1 - max_x_original
                min_y_rot = height - 1 - min_y_original
                max_y_rot = height - 1 - max_y_original
                
                # 旋转后min和max可能互换，需重新排序
                min_x_rot, max_x_rot = sorted([min_x_rot, max_x_rot])
                min_y_rot, max_y_rot = sorted([min_y_rot, max_y_rot])
            else:
                min_x_rot, max_x_rot = min_x_original, max_x_original
                min_y_rot, max_y_rot = min_y_original, max_y_original
            
            # --------------------------
            # 3. 核心：裁剪超出图像边界的坐标（新增步骤）
            # --------------------------
            # 裁剪x坐标到[0, width-1]
            min_x_clamped = np.clip(min_x_rot, 0, width - 1)
            max_x_clamped = np.clip(max_x_rot, 0, width - 1)
            # 裁剪y坐标到[0, height-1]
            min_y_clamped = np.clip(min_y_rot, 0, height - 1)
            max_y_clamped = np.clip(max_y_rot, 0, height - 1)
            
            # 计算裁剪后的有效宽高，过滤无效框
             # 5. 计算有效框面积及占比，过滤小占比框（核心优化）
            valid_width = max_x_clamped - min_x_clamped
            valid_height = max_y_clamped - min_y_clamped
            valid_area = valid_width * valid_height
            area_ratio = valid_area / original_area  # 有效面积占比

            # 框占图像面积占比
            area = width  * height
            ratio = valid_area / area

            MIN_AREA_RATIO = 0.6  # 最小有效面积占比阈值
            # 若有效面积为0 或 占比＜1/5，丢弃该框
            flag = area_ratio < MIN_AREA_RATIO and ratio < MIN_AREA_RATIO
            if valid_area <= 0 or flag:
                # print(f"丢弃小占比框：有效占比={area_ratio:.3f}<{MIN_AREA_RATIO})")
                continue
            
            # 4. 绘制带颜色的矩形框（使用裁剪后的坐标）
            box_color = class_color_map.get(cube['label'], (128, 128, 128))  # 类别颜色
            class_name = cube['label']

            pt1 = (int(min_x_clamped), int(min_y_clamped))  # 裁剪后左上角
            pt2 = (int(max_x_clamped), int(max_y_clamped))  # 裁剪后右下角
            cv2.rectangle(img, pt1, pt2, box_color, 2)  # 仅绘制图像内的部分
            
            # 5. 绘制类别名称（适配裁剪后的位置）
            # 文本位置：裁剪后框的左上角上方（避免超出图像）
            if pt1[1] - 10 > 0:  # 上方有空间
                text_pos = (pt1[0], pt1[1] - 10)
            else:  # 上方无空间，放框内
                text_pos = (pt1[0], pt1[1] + 20)
            
            # 白色文本+黑色描边，确保清晰
            cv2.putText(img, class_name, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)  # 描边
            cv2.putText(img, class_name, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)  # 文本
            
            # # 6. 生成裁剪后的YOLO标签（基于有效区域）
            # # 计算裁剪后框的中心坐标（像素级）
            # x_center_pixel = min_x_clamped + valid_width / 2.0
            # y_center_pixel = min_y_clamped + valid_height / 2.0
            
            # # 归一化（确保在[0,1]范围内）
            # x_center = x_center_pixel / width
            # y_center = y_center_pixel / height
            # w_norm = valid_width / width
            # h_norm = valid_height / height
            
            # # 再次裁剪归一化值（保险措施）
            # x_center = np.clip(x_center, 0.0, 1.0)
            # y_center = np.clip(y_center, 0.0, 1.0)
            # w_norm = np.clip(w_norm, 0.0, 1.0)
            # h_norm = np.clip(h_norm, 0.0, 1.0)
            
            # 添加到YOLO标签列表
            yolo_line = f"{cube['label']} {pt1[0]} {pt1[1]} {pt2[0]} {pt2[1]}"
            yolo_labels.append(yolo_line)
            


    label_path = os.path.join(label_dir, image_name.replace("png","txt"))  # 标签文件路径
    print(f"标签文件路径: {label_path}")
    # 写入标签（若没有目标则创建空文件）
    with open(label_path, 'w', encoding='utf-8') as f:
        if yolo_labels:
            f.write('\n'.join(yolo_labels) + '\n')  # 每行一个目标
        else:
            f.write('')  # 无目标则为空文件


    if image_name in ["00001.png","00100.png"]:
        ## 显示图像（可选）
        cv2.imshow("2D Bounding Boxes", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img

# 6. 主函数
def main(json_path, camera_txt_path, image_folder,save_label_dir,is_rotated_180):
    # 解析数据
    cubes = parse_3d_annotations(json_path)
    cameras = parse_camera_extrinsics(camera_txt_path)
    K, _, _ = get_camera_intrinsics()
    
    print(f"解析到 {len(cubes)} 个3D立方体")
    print(f"解析到 {len(cameras)} 个相机参数")
    print(f"图像文件夹: {image_folder}")
    
    # 为每个相机可视化投影结果
    for image_id in cameras:
        fig = visualize_projection_on_image(cubes, cameras, K, image_folder, image_id,save_label_dir,is_rotated_180)
        if True:
            # if image_id >= 71:
            #     plt.show()
            
            # 可选：保存带投影的图像
            vis_save_dir =  image_folder.replace("fixed_images","fixed_images_with_label")
            os.makedirs(vis_save_dir, exist_ok=True)

            output_path = os.path.join(vis_save_dir,cameras[image_id]['image_name'])
            
            # fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
            cv2.imwrite(output_path, fig)
            print(f"带投影的图像已保存到 {output_path}")





if __name__ == "__main__":

    is_rotated_180 = True

    ai_label_dir = "/mnt/dln/data/datasets/0915/make_label_raw/label/for_2d"
    raw_data_dir = "/mnt/dln/data/datasets/0915/make_label_raw/"
    json_annotation_list = os.listdir(ai_label_dir)
    for json_annotation in json_annotation_list:
        json_annotation_path = os.path.join(ai_label_dir,json_annotation)
        scene_name = json_annotation[:-11]
        camera_extrinsics_path = os.path.join(raw_data_dir,scene_name,"sparse/0/images.txt")
        image_folder_path = os.path.join(raw_data_dir,scene_name,"fixed_images/")
        save_label_dir = os.path.join(raw_data_dir,scene_name,"2D_label/")
        if not os.path.exists(save_label_dir):
            os.makedirs(save_label_dir)
        if len(os.listdir(save_label_dir))!=0:
            continue

        # 检查文件是否存在
        main(json_annotation_path, camera_extrinsics_path, image_folder_path,save_label_dir,is_rotated_180)
    
