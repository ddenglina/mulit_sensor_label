import pickle

# 1. 从 .pkl 文件读取并恢复对象（使用 pickle.load()）
# 打开文件（需指定模式为 "rb"：二进制读取）
pkl_path="/mnt/dln/projects/perception_fusion/data/vipe/metal_earphone/vipe/metal_earphone_info.pkl"
with open(pkl_path, "rb") as f:
    # 读取顺序需与序列化时一致（此处恢复两个对象）
    data= pickle.load(f)
    print(data)