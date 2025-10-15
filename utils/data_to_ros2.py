import rclpy
from rclpy.serialization import serialize_message
from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions, TopicMetadata
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header  # 导入Header消息类型
from sensor_msgs_py import point_cloud2
from cv_bridge import CvBridge
import cv2
import os
import numpy as np
from plyfile import PlyData, PlyElement
from glob import glob
import argparse

def data_to_bag(image_dir, pointcloud_dir, bag_path, 
                image_topic="/camera/image_raw", pointcloud_topic="/lidar/pointcloud",
                image_frame_id="camera_link", pointcloud_frame_id="lidar_link",
                frequency=10):
    """
    将图片和PLY点云文件夹转换为ROS2 bag文件
    """
    # 初始化ROS2
    rclpy.init()
    
    # 创建CVBridge用于图像转换
    bridge = CvBridge()
    
    # 获取文件夹中所有图片和点云的路径
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob(os.path.join(image_dir, ext)))
    
    pointcloud_paths = glob(os.path.join(pointcloud_dir, '*.ply'))
    
    # 按文件名排序
    image_paths.sort()
    pointcloud_paths.sort()
    
    # 检查数据
    if not image_paths:
        print(f"错误: 在 {image_dir} 中未找到任何图片")
        return
    
    if not pointcloud_paths:
        print(f"错误: 在 {pointcloud_dir} 中未找到任何PLY点云文件")
        return
    
    # 确保数据数量匹配
    min_count = min(len(image_paths), len(pointcloud_paths))
    print(f"找到 {len(image_paths)} 张图片和 {len(pointcloud_paths)} 个PLY点云，将处理其中 {min_count} 组数据...")
    
    # 初始化bag写入器
    storage_options = StorageOptions(
        uri=bag_path,
        storage_id="sqlite3"
    )
    
    converter_options = ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr"
    )
    
    writer = SequentialWriter()
    writer.open(storage_options, converter_options)
    
    # 创建话题
    image_topic_metadata = TopicMetadata(
        name=image_topic,
        type="sensor_msgs/msg/Image",
        serialization_format="cdr"
    )
    writer.create_topic(image_topic_metadata)
    
    pointcloud_topic_metadata = TopicMetadata(
        name=pointcloud_topic,
        type="sensor_msgs/msg/PointCloud2",
        serialization_format="cdr"
    )
    writer.create_topic(pointcloud_topic_metadata)
    
    # 计算时间间隔
    time_interval_ns = int(1e9 / frequency)
    
    # 写入数据到bag
    for i in range(min_count):
        timestamp_ns = i * time_interval_ns
        current_time = rclpy.time.Time(nanoseconds=timestamp_ns).to_msg()
        
        # 处理并写入图像
        img_path = image_paths[i]
        cv_image = cv2.imread(img_path)
        if cv_image is None:
            print(f"警告: 无法读取图片 {img_path}，已跳过")
            continue
        
        ros_image = bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
        ros_image.header.frame_id = image_frame_id
        ros_image.header.stamp = current_time
        
        writer.write(
            image_topic,
            serialize_message(ros_image),
            timestamp_ns
        )
        
        # 处理并写入PLY点云（修复Header错误）
        ply_path = pointcloud_paths[i]
        try:
            # 读取PLY文件
            ply_data = PlyData.read(ply_path)
            
            # 查找顶点元素
            vertex_element = None
            for element in ply_data.elements:
                if element.name == 'vertex':
                    vertex_element = element
                    break
            
            if vertex_element is None:
                print(f"警告: PLY文件 {ply_path} 中未找到顶点数据")
                continue
            
            # 提取点云坐标
            x = vertex_element.data['x']
            y = vertex_element.data['y']
            z = vertex_element.data['z']
            
            # 检查是否有颜色信息
            has_color = False
            r, g, b = None, None, None
            if 'red' in vertex_element.data.dtype.names:
                has_color = True
                r = vertex_element.data['red']
                g = vertex_element.data['green']
                b = vertex_element.data['blue']
            elif 'r' in vertex_element.data.dtype.names:
                has_color = True
                r = vertex_element.data['r']
                g = vertex_element.data['g']
                b = vertex_element.data['b']
            
            # 构建点云数据列表
            points = []
            for j in range(len(vertex_element.data)):
                point = [x[j], y[j], z[j]]
                if has_color:
                    # 确保颜色值在0-255范围内并转换为整数
                    point.extend([
                        int(np.clip(r[j], 0, 255)),
                        int(np.clip(g[j], 0, 255)),
                        int(np.clip(b[j], 0, 255))
                    ])
                points.append(point)
            
            # 定义点云字段
            fields = [
                PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1)
            ]
            
            # 如果有颜色信息，添加颜色字段
            if has_color:
                fields.extend([
                    PointField(name="r", offset=12, datatype=PointField.UINT8, count=1),
                    PointField(name="g", offset=13, datatype=PointField.UINT8, count=1),
                    PointField(name="b", offset=14, datatype=PointField.UINT8, count=1)
                ])
            
            # 创建正确的Header对象（关键修复）
            header = Header()
            header.stamp = current_time
            header.frame_id = pointcloud_frame_id
            
            # 转换为ROS PointCloud2消息 - 使用正确的header
            ros_cloud = point_cloud2.create_cloud(header, fields, points)
            
            # 写入bag
            writer.write(
                pointcloud_topic,
                serialize_message(ros_cloud),
                timestamp_ns
            )
            
        except Exception as e:
            print(f"警告: 处理点云 {ply_path} 时出错: {str(e)}，已跳过")
            continue
        
        # 显示进度
        if (i + 1) % 10 == 0 or (i + 1) == min_count:
            print(f"已处理 {i + 1}/{min_count} 组数据")
    
    # 关闭ROS2
    rclpy.shutdown()
    print(f"转换完成! ROS2 bag已保存至: {bag_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='将图片和PLY点云文件夹转换为ROS2 bag文件')
    parser.add_argument('image_dir', help='包含图片的文件夹路径')
    parser.add_argument('pointcloud_dir', help='包含PLY点云文件的文件夹路径')
    parser.add_argument('--output', default='data_bag', help='输出的bag文件名称(默认:data_bag)')
    parser.add_argument('--image_topic', default='/camera/image_raw', help='图像话题名称(默认:/camera/image_raw)')
    parser.add_argument('--pointcloud_topic', default='/lidar/pointcloud', help='点云话题名称(默认:/lidar/pointcloud)')
    parser.add_argument('--image_frame', default='camera_link', help='图像坐标系ID(默认:camera_link)')
    parser.add_argument('--pointcloud_frame', default='lidar_link', help='点云坐标系ID(默认:lidar_link)')
    parser.add_argument('--frequency', type=int, default=10, help='发布频率(Hz)(默认:10)')
    
    args = parser.parse_args()
    
    data_to_bag(
        image_dir=args.image_dir,
        pointcloud_dir=args.pointcloud_dir,
        bag_path=args.output,
        image_topic=args.image_topic,
        pointcloud_topic=args.pointcloud_topic,
        image_frame_id=args.image_frame,
        pointcloud_frame_id=args.pointcloud_frame,
        frequency=args.frequency
    )