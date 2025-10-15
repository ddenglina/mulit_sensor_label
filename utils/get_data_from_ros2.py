
import os
import sys
import cv2
import rclpy
import numpy as np
import open3d as o3d
from cv_bridge import CvBridge
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions


def extract_images_from_bag(db_file, image_topic, output_dir, img_type="rgb"):
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    rclpy.init()
    # 配置存储选项
    storage_options = StorageOptions(
        uri=db_file,  # 替换为你的.db3文件路径
        storage_id='sqlite3'
    )

    # 配置转换器选项
    converter_options = ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )

    # 创建读取器
    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    # 打印所有话题信息
    topic_types = reader.get_all_topics_and_types()
    for topic_type in topic_types:
        print(f"Topic: {topic_type.name}, Type: {topic_type.type}")

    topic_types = reader.get_all_topics_and_types()
    type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}
    msg_type = get_message(type_map[image_topic])

    bridge = CvBridge()
    global_counter = 0
    save_iter = 1

    # 读取并打印消息
    while reader.has_next():
        (topic, data, t) = reader.read_next()
        if topic == image_topic:
            try:
                msg = deserialize_message(data, msg_type)
                cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')  # 'bgr8', 'rgb8'
                image_filename = os.path.join(output_dir, f"image_{global_counter}.png")
                if global_counter % save_iter == 0:
                    # 保存图像到本地
                    cv2.imwrite(image_filename, cv_image)
                    print(f"Saved image: {image_filename}")
                global_counter += 1

            except Exception as e:
                print(f"Error converting image: {e}")
    rclpy.shutdown()
    return


def parse_pointcloud2(msg):
    """解析sensor_msgs/msg/PointCloud2消息为点云数据（修复内存连续性问题）"""
    # 提取字段信息（x,y,z,intensity等）
    fields = {field.name: (field.offset, field.datatype) for field in msg.fields}
    point_step = msg.point_step  # 每个点的字节数
    data = np.frombuffer(msg.data, dtype=np.uint8)  # 原始二进制数据（1维数组）

    # 重塑为二维数组（点数×每个点的字节数）
    data_2d = data.reshape(-1, point_step)

    # 提取x,y,z坐标（确保内存连续）
    def get_contiguous_float32(data_2d, offset):
        """提取指定偏移量的4字节数据，并转换为连续的float32数组"""
        slice_ = data_2d[:, offset:offset+4]  # 列切片（可能非连续）
        contiguous_slice = np.ascontiguousarray(slice_)  # 转换为C连续内存
        return contiguous_slice.view(np.float32).flatten()  # 安全view

    # 解析x,y,z坐标
    x = get_contiguous_float32(data_2d, fields["x"][0])
    y = get_contiguous_float32(data_2d, fields["y"][0])
    z = get_contiguous_float32(data_2d, fields["z"][0])

    # 解析强度字段（可选）
    intensity = None
    for intensity_field in ["intensity", "reflectivity"]:
        if intensity_field in fields:
            offset, dtype = fields[intensity_field]
            # 根据数据类型提取（确保连续）
            if dtype == 2:  # uint8（1字节）
                intensity_slice = data_2d[:, offset:offset+1]
                intensity = np.ascontiguousarray(intensity_slice).view(np.uint8).flatten().astype(np.float32) / 255.0
            elif dtype == 7:  # uint32（4字节）
                intensity_slice = data_2d[:, offset:offset+4]
                intensity = np.ascontiguousarray(intensity_slice).view(np.uint32).flatten().astype(np.float32) / 255.0
            break

    # 组合点云数据
    points = np.column_stack((x, y, z))
    colors = np.column_stack((intensity, intensity, intensity)) if intensity is not None else None
    return points, colors


def extract_lidar_from_bag(db_file, lidar_topic, output_dir):
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    rclpy.init()
    # 配置存储选项
    storage_options = StorageOptions(
        uri=db_file,  # 替换为你的.db3文件路径
        storage_id='sqlite3'
    )

    # 配置转换器选项
    converter_options = ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )

    # 创建读取器
    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    # 打印所有话题信息
    topic_types = reader.get_all_topics_and_types()
    for topic_type in topic_types:
        print(f"Topic: {topic_type.name}, Type: {topic_type.type}")

    topic_types = reader.get_all_topics_and_types()
    type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}
    msg_type = get_message(type_map[lidar_topic])

    global_counter = 0
    save_iter = 1

    # 读取并打印消息
    while reader.has_next():
        (topic, data, t) = reader.read_next()
        if topic == lidar_topic:
            # try:
            msg = deserialize_message(data, msg_type)

            # 转换为点云数据
            points, colors = parse_pointcloud2(msg)

            # 创建并保存点云
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            if colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(colors)

            # 创建Open3D点云对象并保存
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            # pcd.colors = o3d.utility.Vector3dVector(np.tile(points[:, 3:], 3))  # 强度转灰度

            # 生成点云文件名
            pcd_filename = os.path.join(output_dir, f"image_{global_counter}.pcd")

            if global_counter % save_iter == 0:
                # 保存点云到本地
                o3d.io.write_point_cloud(pcd_filename, pcd, write_ascii=True)
                print(f"Saved point cloud: {pcd_filename}")
            global_counter += 1

            # except Exception as e:
            #     print(f"Error converting image: {e}")
    rclpy.shutdown()

    return


def main():

    # # image1
    # db_file = r'./rosbag2_2025_07_30-12_02_39/rosbag2_2025_07_30-12_02_39_0.db3'
    # image_topic = '/camera2/camera2/color/image_raw'  # '/camera/camera/color/image_raw' '/aima/hal/fish_eye_camera/chest_left/color'
    # # root_dir = os.path.splitext(db_file)[0]
    # # output_dir = os.path.join(root_dir, "image")
    # output_dir = os.path.join(os.path.dirname(db_file), "unpacked_data", "unpacked_image")
    # extract_images_from_bag(db_file, image_topic, output_dir, img_type="rgb")

    # # image2
    # db_file = r'./ros2_bag_fast_calib_0611_02_0.db3'
    # image_topic = '/left_stereo_camera/left/image_raw'
    # root_dir = os.path.splitext(db_file)[0]
    # output_dir = os.path.join(root_dir, "image_right_undis")
    # extract_images_from_bag(db_file, image_topic, output_dir, img_type="rgb")

    # lidar
    db_file = r'/mnt/dln/resource/datasets/car_lidar_d455_wct_0813/rosbag2_2025_08_13-18_52_31_0.db3'
    lidar_topic = '/livox/lidar'  # '/livox/lidar'  '/aima/hal/lidar/neck/pointcloud' '/livox/lidar_192_168_1_119'
    # root_dir = os.path.splitext(db_file)[0]
    # output_dir = os.path.join(root_dir, "unpacked_lidar")
    output_dir = os.path.join(os.path.dirname(db_file), "unpacked_data", "unpacked_lidar_143")
    extract_lidar_from_bag(db_file, lidar_topic, output_dir)

    return


if __name__ == '__main__':
    # path = '/root/zhangyuanyi/dataset/rosbag/rosbag2_2025_05_21-14_21_37/rosbag2_2025_05_21-14_21_37_0.db3'
    main()


