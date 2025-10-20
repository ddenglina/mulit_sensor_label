import open3d as o3d
import argparse
import os

def convert_pcd_to_ply(input_path, output_path):
    """
    将单个PCD格式点云文件转换为PLY格式
    """
    try:
        # 读取PCD文件
        pcd = o3d.io.read_point_cloud(input_path)
        
        # 检查点云是否为空
        if len(pcd.points) == 0:
            print(f"警告：{input_path} 是为空点云，已跳过")
            return False
        
        # 保存为PLY文件
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"已转换：{os.path.basename(input_path)} -> {os.path.basename(output_path)}")
        return True
        
    except Exception as e:
        print(f"转换失败 {input_path}：{str(e)}")
        return False

def batch_convert(folder_path, output_folder=None):
    """
    批量转换文件夹中的所有PCD文件
    
    参数:
        folder_path: 包含PCD文件的文件夹路径
        output_folder: 输出PLY文件的文件夹路径，默认为原文件夹
    """
    # 设置输出文件夹，默认为输入文件夹
    if output_folder is None:
        output_folder = folder_path
    
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)
    
    # 统计转换数量
    converted_count = 0
    total_count = 0
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件是否为PCD格式
        if filename.lower().endswith('.pcd'):
            total_count += 1
            input_path = os.path.join(folder_path, filename)
            
            # 生成输出文件名（替换扩展名）
            ply_filename = os.path.splitext(filename)[0] + '.ply'
            output_path = os.path.join(output_folder, ply_filename)
            
            # 执行转换
            if convert_pcd_to_ply(input_path, output_path):
                converted_count += 1
    
    print(f"\n转换完成：共处理 {total_count} 个PCD文件，成功转换 {converted_count} 个")


if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='批量将PCD格式点云转换为PLY格式')
    parser.add_argument('input_folder', help='包含PCD文件的文件夹路径')
    parser.add_argument('-o', '--output_folder', help='输出PLY文件的文件夹路径（可选）')
    
    args = parser.parse_args()
    
    # 验证输入文件夹是否存在
    if not os.path.isdir(args.input_folder):
        print(f"错误：输入路径 {args.input_folder} 不是有效的文件夹")
    else:
        # 执行批量转换
        batch_convert(args.input_folder, args.output_folder)
    