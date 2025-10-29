import os
import shutil
import numpy as np
import cv2
from cv_bridge import CvBridge
import rosbag2_py
from sensor_msgs.msg import Image as SensorImage
from sensor_msgs.msg import PointCloud2
from rclpy.serialization import deserialize_message, serialize_message


def split_bag(input_bag_path, output_dir, split_ratio=0.5):
    """
    将大型ROS2 bag按时间分割为两个子包
    :param input_bag_path: 输入bag路径
    :param output_dir: 输出子包目录
    :param split_ratio: 第一个子包占比(0-1)
    :return: 两个子包的路径列表
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    bag1_path = os.path.join(output_dir, "split_part1")
    bag2_path = os.path.join(output_dir, "split_part2")
    
    # 清理可能存在的旧文件
    if os.path.exists(bag1_path):
        shutil.rmtree(bag1_path)
    if os.path.exists(bag2_path):
        shutil.rmtree(bag2_path)

    # 读取输入bag信息
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=input_bag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr"
    )
    reader.open(storage_options, converter_options)

    # 获取所有话题类型
    topic_types = reader.get_all_topics_and_types()
    topic_type_map = {t.name: t.type for t in topic_types}

    # 收集所有消息的时间戳以确定分割点
    all_timestamps = []
    while reader.has_next():
        _, _, timestamp = reader.read_next()
        all_timestamps.append(timestamp)
    
    if not all_timestamps:
        raise ValueError("输入bag文件中没有找到任何消息")
    
    # 确定分割时间点
    all_timestamps.sort()
    split_index = int(len(all_timestamps) * split_ratio)
    split_timestamp = all_timestamps[split_index]
    print(f"数据包分割时间点: {split_timestamp}ns (约{split_timestamp/1e9:.2f}秒)")

    # 重置阅读器
    reader.reset()

    # 创建两个写入器
    writer1 = rosbag2_py.SequentialWriter()
    writer1.open(rosbag2_py.StorageOptions(uri=bag1_path, storage_id="sqlite3"), 
                rosbag2_py.ConverterOptions())
    
    writer2 = rosbag2_py.SequentialWriter()
    writer2.open(rosbag2_py.StorageOptions(uri=bag2_path, storage_id="sqlite3"), 
                rosbag2_py.ConverterOptions())

    # 为写入器创建话题
    for topic in topic_types:
        writer1.create_topic(topic)
        writer2.create_topic(topic)

    # 分割消息到两个子包
    count1, count2 = 0, 0
    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        if timestamp <= split_timestamp:
            writer1.write(topic, data, timestamp)
            count1 += 1
        else:
            writer2.write(topic, data, timestamp)
            count2 += 1

    print(f"数据包分割完成:")
    print(f"  子包1: {bag1_path} ({count1}条消息)")
    print(f"  子包2: {bag2_path} ({count2}条消息)")
    return [bag1_path, bag2_path]


def get_lidar_timestamps(bag_path):
    """从bag文件的/synced/livox/lidar话题中获取激光雷达时间戳列表"""
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr"
    )
    reader.open(storage_options, converter_options)

    # 检查激光雷达话题是否存在
    topic_types = reader.get_all_topics_and_types()
    lidar_topic = "/synced/livox/lidar"
    existing_topics = [topic.name for topic in topic_types]
    if lidar_topic not in existing_topics:
        raise ValueError(f"ROS2 bag中未找到激光雷达话题：{lidar_topic}")

    # 提取激光雷达时间戳
    lidar_timestamps = []
    while reader.has_next():
        topic, data, timestamp_ns = reader.read_next()
        if topic == lidar_topic:
            # 验证消息类型
            try:
                deserialize_message(data, PointCloud2())
                lidar_timestamps.append(timestamp_ns)
            except Exception as e:
                print(f"解析激光雷达消息失败: {e}")

    print(f"从bag {os.path.basename(bag_path)} 中提取到 {len(lidar_timestamps)} 个激光雷达时间戳")
    return lidar_timestamps


def load_bag_image_messages(bag_path, image_topics):
    """加载多个图像话题的消息并按话题存储，同时预处理时间戳数组"""
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr"
    )
    reader.open(storage_options, converter_options)

    # 校验所有图像话题是否存在
    topic_types = reader.get_all_topics_and_types()
    existing_topics = [topic.name for topic in topic_types]
    for topic in image_topics:
        if topic not in existing_topics:
            raise ValueError(f"ROS2 bag中未找到图像话题：{topic}")

    # 按话题存储图像消息
    topic_images = {topic: {"ts": [], "msgs": []} for topic in image_topics}
    while reader.has_next():
        topic, data, timestamp_ns = reader.read_next()
        if topic in topic_images:
            img_msg = deserialize_message(data, SensorImage())
            topic_images[topic]["ts"].append(timestamp_ns)
            topic_images[topic]["msgs"].append(img_msg)

    # 转换为numpy数组便于快速查找
    for topic in image_topics:
        topic_images[topic]["ts_np"] = np.array(topic_images[topic]["ts"], dtype=np.int64)
        print(f"从bag {os.path.basename(bag_path)} 中加载 {topic} 消息 {len(topic_images[topic]['ts'])} 条")
    return topic_images


def find_closest_image(lidar_ts_ns, image_ts_np):
    """找到与激光雷达时间戳最接近的图像时间戳（二分查找优化）"""
    idx = np.searchsorted(image_ts_np, lidar_ts_ns, side='left')
    candidates = []
    if idx > 0:
        candidates.append(idx - 1)
    if idx < len(image_ts_np):
        candidates.append(idx)
    if not candidates:
        return None, None
    
    diffs = np.abs(image_ts_np[candidates] - lidar_ts_ns)
    min_idx = candidates[np.argmin(diffs)]
    return image_ts_np[min_idx], diffs[np.argmin(diffs)]


def save_image_from_msg(img_msg, save_path, encoding="bgr8"):
    """将图像消息转换为OpenCV格式并保存"""
    bridge = CvBridge()
    try:
        if img_msg.encoding == 'mono8':
            cv_img = bridge.imgmsg_to_cv2(img_msg, desired_encoding='mono8')
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
        else:
            cv_img = bridge.imgmsg_to_cv2(img_msg, desired_encoding=encoding)
        
        cv_img = cv2.resize(cv_img, (512, 384))
        cv2.imwrite(save_path, cv_img)
        return True
    except Exception as e:
        print(f"图像转换/保存失败：{e}")
        return False


def process_single_bag(bag_path, save_root, part_num, image_topics, max_time_diff_ns=100*10**6):
    """处理单个子包，提取图像并保存"""
    part_save_root = os.path.join(save_root, f"part_{part_num}")
    print(f"\n处理子包 {part_num}，保存路径: {part_save_root}")

    # 初始化保存目录
    if os.path.exists(part_save_root):
        shutil.rmtree(part_save_root)
    for topic in image_topics:
        cam_name = topic.split('/')[2]
        cam_dir = os.path.join(part_save_root, cam_name)
        os.makedirs(cam_dir, exist_ok=True)

    try:
        # 获取激光雷达时间戳
        lidar_timestamps = get_lidar_timestamps(bag_path)
        if not lidar_timestamps:
            print(f"子包 {part_num} 未提取到任何激光雷达时间戳，跳过")
            return

        # 加载图像消息
        topic_images = load_bag_image_messages(bag_path, image_topics)
    except Exception as e:
        print(f"处理子包 {part_num} 失败：{e}")
        return

    # 匹配并保存图像
    for idx, lidar_ts_ns in enumerate(lidar_timestamps):
        for topic in image_topics:
            cam_name = topic.split('/')[2]
            img_data = topic_images[topic]
            if len(img_data["ts_np"]) == 0:
                print(f"话题 {topic} 无图像消息，跳过")
                continue

            # 查找最近的图像
            closest_img_ts, diff_ns = find_closest_image(lidar_ts_ns, img_data["ts_np"])
            if closest_img_ts is None:
                print(f"话题 {topic} 无有效图像时间戳，跳过")
                continue

            # 检查时间差
            if diff_ns > max_time_diff_ns:
                print(f"激光雷达时间戳与{topic}时间差过大（{diff_ns/1e6:.2f}ms），跳过")
                continue

            # 保存图像
            img_filename = f"lidar_part{part_num}_{idx:05d}.png"
            save_path = os.path.join(part_save_root, cam_name, img_filename)
            
            ts_idx = np.where(img_data["ts_np"] == closest_img_ts)[0][0]
            img_msg = img_data["msgs"][ts_idx]
            if save_image_from_msg(img_msg, save_path):
                print(f"保存 {cam_name} 图像 {idx+1}/{len(lidar_timestamps)} "
                      f"(时间差: {diff_ns/1e6:.2f}ms)")


def main():
    # 配置参数
    root_dir = "/mnt/dln/data/datasets/0915/for_bev_fisheye_test/test2bag_lidar/"
    save_root = os.path.join(root_dir, "Colmap_split")  # 分割后的图像保存根目录
    original_bag_path = "/mnt/dln/data/datasets/bev_28#_1021/123/test2/"  # 原始大型bag路径
    split_output_dir = os.path.join(root_dir, "split_bags")  # 分割后的子包保存目录
    split_ratio = 0.5  # 分割比例，0.5表示平均分成两个包
    max_time_diff_ns = 100 * 10**6  # 最大允许时间差（100ms）

    # 鱼眼相机话题列表
    fisheye_topics = [
        "/fisheye/bleft/image_raw",
        "/fisheye/right/image_raw",
        "/fisheye/bright/image_raw",
        "/fisheye/left/image_raw"
    ]

    try:
        # 分割大型数据包为两个子包
        print("开始分割大型数据包...")
        sub_bags = split_bag(original_bag_path, split_output_dir, split_ratio)
        
        # 分别处理每个子包
        for i, bag_path in enumerate(sub_bags, 1):
            process_single_bag(bag_path, save_root, i, fisheye_topics, max_time_diff_ns)

        print("\n所有子包处理完成")
        print(f"分割后的子包保存于: {split_output_dir}")
        print(f"提取的图像保存于: {save_root}")

    except Exception as e:
        print(f"程序执行失败: {e}")


if __name__ == "__main__":
    main()
