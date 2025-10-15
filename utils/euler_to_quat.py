import numpy as np
from scipy.spatial.transform import Rotation as R

# 原始数据
rotationX = 0.0
rotationY = 0.0
rotationZ = 0.11780972450961684  # 欧拉角（弧度）
quat = [0.0, 0.0, 0.05887080365118883, 0.9982656101847159]  # 四元数 [x, y, z, w]

# 1. 将欧拉角转换为四元数
# 假设旋转顺序为XYZ（需要根据实际情况确认旋转顺序）
r_from_euler = R.from_euler('xyz', [rotationX, rotationY, rotationZ])
quat_from_euler = r_from_euler.as_quat()  # 输出 [x, y, z, w]

# 2. 将四元数转换为欧拉角
r_from_quat = R.from_quat(quat)
euler_from_quat = r_from_quat.as_euler('xyz')  # 输出 [x, y, z] 弧度

# 打印结果进行比较
print("原始欧拉角 (X, Y, Z):")
print(f"[{rotationX:.6f}, {rotationY:.6f}, {rotationZ:.6f}]")
print("\n从欧拉角转换得到的四元数 (x, y, z, w):")
print(f"[{quat_from_euler[0]:.6f}, {quat_from_euler[1]:.6f}, {quat_from_euler[2]:.6f}, {quat_from_euler[3]:.6f}]")
print("\n原始四元数 (x, y, z, w):")
print(f"[{quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}]")
print("\n从四元数转换得到的欧拉角 (X, Y, Z):")
print(f"[{euler_from_quat[0]:.6f}, {euler_from_quat[1]:.6f}, {euler_from_quat[2]:.6f}]")

# 验证误差
print("\n转换误差:")
print(f"四元数误差: {np.linalg.norm(quat_from_euler - quat):.10f}")
print(f"欧拉角误差: {np.linalg.norm([rotationX, rotationY, rotationZ] - euler_from_quat):.10f}")

# 验证数学关系（绕Z轴旋转的特殊情况）
theta = rotationZ  # 绕Z轴旋转角度
print("\n绕Z轴旋转的理论验证:")
print(f"理论四元数 w = cos(theta/2) = {np.cos(theta/2):.6f}")
print(f"理论四元数 z = sin(theta/2) = {np.sin(theta/2):.6f}")
