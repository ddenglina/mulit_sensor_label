import cv2
import os
import glob

def images_to_mp4(image_folder, output_video, fps=30, size=None):
    """
    将文件夹中的图片转换为MP4视频
    
    参数:
        image_folder (str): 包含图片的文件夹路径
        output_video (str): 输出视频的文件路径
        fps (int): 视频帧率，默认30帧/秒
        size (tuple): 视频尺寸 (宽度, 高度)，默认使用第一张图片的尺寸
    """
    # 获取文件夹中所有图片文件（按文件名排序）
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_folder, ext)))
    
    # 按文件名排序（确保图片顺序正确）
    image_files.sort()
    
    if not image_files:
        print("错误：指定文件夹中没有找到图片文件")
        return
    
    # 读取第一张图片获取尺寸
    first_image = cv2.imread(image_files[0])
    if first_image is None:
        print(f"错误：无法读取图片 {image_files[0]}")
        return
    
    # 设置视频尺寸
    if size is None:
        height, width = first_image.shape[:2]
        size = (width, height)
    else:
        width, height = size
    
    # 定义视频编码器和创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4编码器
    out = cv2.VideoWriter(output_video, fourcc, fps, size)
    
    if not out.isOpened():
        print("错误：无法创建视频文件，请检查输出路径和编码器")
        return
    
    # 逐帧处理图片并写入视频
    for image_file in image_files:
        img = cv2.imread(image_file)
        if img is None:
            print(f"警告：跳过无法读取的图片 {image_file}")
            continue
        
        # 调整图片尺寸以匹配视频尺寸
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, size)
        
        out.write(img)
        print(f"已处理: {image_file}")
    
    # 释放资源
    out.release()
    cv2.destroyAllWindows()
    
    print(f"视频已成功生成: {output_video}")

# 使用示例
if __name__ == "__main__":
    # 设置图片文件夹路径
    image_folder = "/mnt/dln/projects/perception_fusion/data/Colmap/traj/renders/"  # 替换为你的图片文件夹路径
    
    # 设置输出视频路径和文件名
    output_video = "/mnt/dln/projects/perception_fusion/data/Colmap/traj/renders/traj.mp4"  # 输出视频的保存路径
    
    # 转换图片为视频
    images_to_mp4(
        image_folder=image_folder,
        output_video=output_video,
        fps=24,  # 帧率设置为24帧/秒
        # size=(1280, 720)  # 可选：指定视频尺寸，如不指定则使用第一张图片的尺寸
    )
    