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



if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='将PCD格式点云转换为PLY格式（支持单个文件或目录批量处理）')
    parser.add_argument('input_path', help='输入路径')
    parser.add_argument('-o', '--output', help='输出路径')
    
    args = parser.parse_args()
    
    # 处理输入路径
    input_path = args.input_path
    output_path = args.output
   
    convert_pcd_to_ply(input_path, output_path)
        
