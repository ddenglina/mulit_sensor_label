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
    parser = argparse.ArgumentParser(description='将PCD格式点云转换为PLY格式（支持单个文件或目录批量处理）')
    parser.add_argument('input_path', help='输入路径（可以是PCD文件或包含PCD文件的文件夹）')
    parser.add_argument('-o', '--output', help='输出路径（可选，若输入为文件则为输出文件路径，若输入为目录则为输出目录路径）')
    
    args = parser.parse_args()
    
    # 处理输入路径
    input_path = args.input_path
    
    # 检查输入路径是否存在
    if not os.path.exists(input_path):
        print(f"错误：输入路径 {input_path} 不存在")
    else:
        # 处理单个文件
        if os.path.isfile(input_path):
            # 检查是否为PCD文件
            if not input_path.lower().endswith('.pcd'):
                print(f"错误：{input_path} 不是PCD文件")
            else:
                # 处理输出路径
                if args.output:
                    output_path = args.output
                    # 确保输出目录存在
                    output_dir = os.path.dirname(output_path)
                    if output_dir and not os.path.exists(output_dir):
                        os.makedirs(output_dir, exist_ok=True)
                else:
                    # 默认为原文件同目录下的PLY文件
                    output_path = os.path.splitext(input_path)[0] + '.ply'
                
                # 执行单个文件转换
                convert_pcd_to_ply(input_path, output_path)
        
        # 处理目录（批量转换）
        elif os.path.isdir(input_path):
            batch_convert(input_path, args.output)
