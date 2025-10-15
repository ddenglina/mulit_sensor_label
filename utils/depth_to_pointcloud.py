
import OpenEXR
import Imath
import numpy as np
import cv2
import open3d as o3d

def read_exr_channel(exr_file_path, channel='Z'):
    """读取EXR文件中的指定通道数据"""
    # 打开EXR文件
    exr_file = OpenEXR.InputFile(exr_file_path)
    
    # 获取图像尺寸
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    
    # 检查通道是否存在
    if channel not in header['channels']:
        raise ValueError(f"EXR文件中不存在 {channel} 通道")
    
    # 定义像素类型为32位浮点数
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    
    # 读取通道数据并转换为numpy数组
    channel_str = exr_file.channel(channel, pt)
    channel_data = np.frombuffer(channel_str, dtype=np.float32)
    channel_data = channel_data.reshape((height, width))
    
    return channel_data

def create_colored_pointcloud(exr_depth_path, color_image_path, intrinsics, 
                             depth_scale=1.0, max_depth=None):
    """
    从EXR深度图和彩色图像创建带颜色的点云
    
    参数:
        exr_depth_path: EXR格式深度图路径
        color_image_path: 彩色图像路径
        intrinsics: 相机内参字典，包含'fx', 'fy', 'cx', 'cy'
        depth_scale: 深度缩放因子
        max_depth: 最大深度阈值，超过此值的点将被过滤
    
    返回:
        open3d.geometry.PointCloud 对象
    """
    # 1. 读取EXR深度通道
    depth_map = read_exr_channel(exr_depth_path, 'Z')
    height, width = depth_map.shape
    
    # 2. 读取并处理彩色图像
    color_image = cv2.imread(color_image_path)
    if color_image is None:
        raise FileNotFoundError(f"无法读取彩色图像: {color_image_path}")
    
    # 调整彩色图像尺寸以匹配深度图
    if color_image.shape[:2] != (height, width):
        color_image = cv2.resize(color_image, (width, height), interpolation=cv2.INTER_AREA)
    
    # 转换为RGB格式（OpenCV默认读取为BGR）
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    
    # 3. 创建像素坐标网格
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    u = u.flatten()
    v = v.flatten()
    
    # 4. 处理深度值
    z = depth_map.flatten() * depth_scale
    
    # 过滤无效深度值
    valid_mask = z > 1e-6  # 排除零或负值深度
    
    # 应用最大深度阈值（如果提供）
    if max_depth is not None:
        valid_mask = np.logical_and(valid_mask, z <= max_depth)
    
    # 应用过滤
    u = u[valid_mask]
    v = v[valid_mask]
    z = z[valid_mask]
    
    # 5. 计算三维坐标
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']
    
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # 6. 获取对应像素的颜色
    # 将图像展平并应用相同的过滤
    color_flat = color_image.reshape(-1, 3)[valid_mask]
    # 归一化到0-1范围
    color_normalized = color_flat / 255.0
    
    # 7. 创建点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.column_stack([x, y, z]))
    pcd.colors = o3d.utility.Vector3dVector(color_normalized)
    
    return pcd

def main():
    # 配置参数
    exr_depth_path = "/mnt/dln/data/vipe/vipe/vipe_results/depth/00020.exr"       # EXR深度图路径
    color_image_path = "/mnt/dln/data/vipe/vipe/vipe_results/rgb/images/frame_000020.jpg"     # 彩色图像路径
    output_pointcloud_path = "/mnt/dln/data/vipe/vipe/vipe_results/depth/00020_rgb_pointcloud.ply"  # 输出点云路径
    
    # 相机内参（根据实际相机参数修改）
    intrinsics = {
        'fx': 571.65155,   # 焦距x
        'fy': 571.65155,   # 焦距y
        'cx': 320,   # 主点x
        'cy': 240    # 主点y
    }
    
    # 深度缩放因子和最大深度（根据实际场景调整）
    depth_scale = 1.0    # 如果EXR中的Z值单位是米，则为1.0
    max_depth = 20.0      # 过滤超过5米的点（根据需要调整或设为None）
    
    try:
        # 生成带颜色的点云
        print("正在生成带颜色的点云...")
        pcd = create_colored_pointcloud(
            exr_depth_path, 
            color_image_path, 
            intrinsics, 
            depth_scale, 
            max_depth
        )
        
        print(f"生成的点云包含 {len(pcd.points)} 个点")
        
        # 保存点云
        o3d.io.write_point_cloud(output_pointcloud_path, pcd)
        print(f"带颜色的点云已保存到: {output_pointcloud_path}")
        
        # 可视化点云
        print("可视化点云... (按ESC键退出)")
        o3d.visualization.draw_geometries([pcd], window_name="带颜色的点云")
        
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")

if __name__ == "__main__":
    main()




