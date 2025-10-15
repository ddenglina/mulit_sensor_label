#!/data/miniconda3/envs/openpcdet/bin/python3
import sys
sys.path.append('/data/miniconda3/envs/openpcdet/lib/python3.10/site-packages') 
sys.path.append('/data/project/OpenPCDet_ziyu') 
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from vision_msgs.msg import Detection3DArray
from vision_msgs.msg import BoundingBox2D, BoundingBox2DArray
from vision_msgs.msg import Detection3D
from vision_msgs.msg import ObjectHypothesisWithPose
from sensor_msgs.msg import Image as ig
from perception_fusion_msg.msg import CustomPerceptionFusion, CustomHuman  # 导入自定义消息
from std_msgs.msg import String, Int32MultiArray
import os
import time
import ros2_numpy
import numpy as np
import torch
from pcdet.datasets import DatasetTemplate
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.models import build_network, load_data_to_gpu
from pyquaternion import Quaternion
from pcdet.utils.box_utils import boxes3d_to_corners3d_kitti_camera
from cv_bridge import CvBridge
import cv2

def draw_projected_box3d(image, qs, color=(255, 255, 0), thickness=3, text=None):
    """ Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    """
    qs = qs.astype(np.int32)
    for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        # use LINE_AA for opencv3
        # cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

    # 新增对于朝向的可视化，不一定可靠
    cv2.line(image, (qs[1, 0], qs[1, 1]), (qs[4, 0], qs[4, 1]), color, thickness)
    cv2.line(image, (qs[0, 0], qs[0, 1]), (qs[5, 0], qs[5, 1]), color, thickness)
    
    if text is not None:
        cv2.putText(image, text, (int((qs[0, 0]+qs[1, 0])/2), int((qs[0, 1]+qs[1, 1])/2-100)), cv2.FONT_HERSHEY_SIMPLEX, 2, (203, 192, 255), 2)
    
    return image

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )

class Subscriber(Node):
    def __init__(self,name):
        super().__init__(name)
        # 订阅人脸检测结果
        self.face_det_sub = self.create_subscription(BoundingBox2DArray, "/wct_face/bbox",self.face_det_callback,10)

        # 订阅人物姓名
        self.face_name_sub = self.create_subscription(String, "/wct_face/name",self.face_name_callback,10)
        
        # 订阅人脸关键点结果
        self.face_kp_sub = self.create_subscription(Int32MultiArray, "/wct_face/landmark", self.face_kp_callback,10)

        # 订阅人脸模块可视化结果
        self.face_vis_sub = self.create_subscription(ig, "/wct_face/vis_image",self.face_vis_callback, 10) 
        
        # 订阅3D跟踪结果
        self.det_3D_sub = self.create_subscription(Detection3DArray, "simpletrack", self.sub_callback, 10)
        
        # 发布3D-2D感知融合图像
        self.human_fusion_vis_pub = self.create_publisher(ig, "/hpe/vis_image", 10)

        # 发布3D-2D感知融合结果
        self.human_fusion_pub = self.create_publisher(CustomPerceptionFusion, "/hpe/human_fusion", 10)

        self.get_logger().info('start node of {}'.format(name))

        # 初始化各投影参数
        self.init_params()

        self.get_logger().info('开启感知融合节点')
 
    def init_params(self):
        # 非广角相机，无需畸变矫正
        self.K = np.array([
            [1383.06, 0, 937.531],
            [0, 1381.96, 492.524],
            [0, 0, 1]
        ], dtype=np.float64)

        self.lidar_2_cam = np.eye(4).astype(np.float32)
        self.lidar_2_cam[:3, :3] = np.array([
            [ 0.003142,  0.999634,  0.026873],
            [-0.074997, -0.026562,  0.996830],
            [ 0.997179, -0.005147,  0.074887]
        ])
        self.lidar_2_cam[:3, 3] = np.array([0.170745, 0.358835, 0.036695])

        self.face_box_list = []
        self.face_name_list = []
        self.face_kp_list = []

        self.x_thre = 0.2

        self.bridge = CvBridge()
        self.face_vis_img = None

        self.num_people = 0

        self.pixel_thre = 80
        

    def face_det_callback(self, msg):
        # 解析人脸检测消息，注意，只根据人脸检测来修改self.num_people
        self.num_people = len(msg.boxes)
        self.face_box_list = []
        if self.num_people > 0:
            for i in range(self.num_people):
                self.face_box_list.append([msg.boxes[i].center.position.x, msg.boxes[i].center.position.y, msg.boxes[i].size_x, msg.boxes[i].size_y]) 
            
        # print (self.face_box_list)

    def face_name_callback(self, msg):
        # 解析人物名字消息
        self.face_name_list = []
        if self.num_people > 0:
            self.face_name_list = msg.data.split(',')
            
        # print (self.face_name_list)

    def face_kp_callback(self, msg):
        # 解析人物关键点消息
        self.face_kp_list = []
        if self.num_people > 0:
            self.face_kp_list = list(msg.data)
        
        # print (self.face_kp_list)

    def face_vis_callback(self, msg):
        self.face_vis_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
    
    def sub_callback(self, msg):
        if  self.face_vis_img is not None:
            vis_img = self.face_vis_img.copy()
        else:
            vis_img = None

        # 0. 保存目前所有人脸相关结果
        cur_face_box = self.face_box_list.copy()
        cur_face_name = self.face_name_list.copy()
        cur_face_kp = self.face_kp_list.copy()
        cur_face_num = self.num_people

        # 1. 从msg解析出所有3D box
        detections = msg.detections

        # 新建返回消息
        hpf_msg = CustomPerceptionFusion()
        hpf_msg.header.frame_id = msg.header.frame_id
        hpf_msg.header.stamp = msg.header.stamp
        hpf_msg.object_num = len(detections)

        # 如果一个3D检测结果都没有，直接发布
        if hpf_msg.object_num < 1:
            self.human_fusion_pub.publish(hpf_msg)
            self.get_logger().info("共看到{}个人和{}个人脸".format(hpf_msg.object_num, cur_face_num))

            if vis_img is not None:
                self.human_fusion_vis_pub.publish(self.bridge.cv2_to_imgmsg(vis_img, encoding="bgr8"))
            return
        
        # 如果一个人脸都没有，直接发布3D的结果即可
        if cur_face_num < 1:
            for det in detections:
                # 提取目标信息（使用临时变量减少属性链访问）
                center = det.bbox.center.position
                size = det.bbox.size
                rotation = det.results[0].hypothesis.class_id.split(':')[-1]
                track_id = det.results[0].hypothesis.score
                cur_human = CustomHuman()
                cur_human.center_x = center.x
                cur_human.center_y = center.y
                cur_human.center_z = center.z
                cur_human.length = size.x
                cur_human.width = size.y
                cur_human.height = size.z
                cur_human.rotation = float(rotation)
                cur_human.track_id = int(track_id)
                hpf_msg.human.append(cur_human)
            
            self.human_fusion_pub.publish(hpf_msg)
            self.get_logger().info("共看到{}个人和{}个人脸".format(hpf_msg.object_num, cur_face_num))

            if vis_img is not None:
                self.human_fusion_vis_pub.publish(self.bridge.cv2_to_imgmsg(vis_img, encoding="bgr8"))
            return
    
        # 2. 既有人脸又有3D，将3D结果与人脸结果进行匹配后发布
        valid_detections = []
        for det in detections:
            # 提取目标信息（使用临时变量减少属性链访问）
            center = det.bbox.center.position
            size = det.bbox.size
            rotation = det.results[0].hypothesis.class_id.split(':')[-1]
            track_id = det.results[0].hypothesis.score

            # 只有在机器人相机FOV内的3D检测结果才参与人脸匹配
            if (det.bbox.center.position.x > self.x_thre) and (np.abs(np.arctan2(center.y, center.z))<=70 / 2): #对于可视化操作，只保留前向一定范围的3D检测结果
                # 存储为元组/列表（比固定数组更灵活）
                valid_detections.append([
                    center.x, center.y, center.z,
                    size.x, size.y, size.z,
                    rotation, float(track_id)
                ])
            
            # 其余结果直接写入CustomHuman中
            else:
                cur_human = CustomHuman()
                cur_human.center_x = center.x
                cur_human.center_y = center.y
                cur_human.center_z = center.z
                cur_human.length = size.x
                cur_human.width = size.y
                cur_human.height = size.z
                cur_human.rotation = float(rotation)
                cur_human.track_id = int(track_id)
                hpf_msg.human.append(cur_human)
        
        if len(valid_detections) < 1:
            self.human_fusion_pub.publish(hpf_msg)
            self.logger.info("共看到{}个人和{}个人脸".format(hpf_msg.object_num, cur_face_num))

            if vis_img is not None:
                self.human_fusion_vis_pub.publish(self.bridge.cv2_to_imgmsg(vis_img, encoding="bgr8"))
            return
        
        det_3D_result = np.array(valid_detections, dtype=float)

        # 将3d检测结果转到相机坐标下
        xyz_lidar = det_3D_result[:, 0:3]
        l, w, h = det_3D_result[:, 3:4], det_3D_result[:, 4:5], det_3D_result[:, 5:6]
        r = det_3D_result[:, 6:7]

        # 移动到bottom center，后面转corners3d要用
        xyz_lidar[:, 2] += h.reshape(-1) / 2

        # 将点云投影到畸变后的图像
        xyz_lidar_homo = np.hstack([xyz_lidar, np.ones(xyz_lidar.shape[0], dtype=np.float32).reshape((-1, 1))])

        xyz_cam_homo = xyz_lidar_homo @ self.lidar_2_cam.T  # 形状[N,4] (N为点数)
        xyz_cam = xyz_cam_homo[:, :3]  # 相机坐标系X,Y,Z（非齐次）
        r = -r - np.pi / 2
        boxes_cam = np.concatenate([xyz_cam, l, h, w, r], axis=-1)
        boxes_cam_corners3d = boxes3d_to_corners3d_kitti_camera(boxes_cam).reshape(-1, 3)
        boxes_cam_corners3d_norm = boxes_cam_corners3d[:, :2] / boxes_cam_corners3d[:, 2:3] 
        boxes_img_corners3d = np.dot(boxes_cam_corners3d_norm, self.K[:2, :2].T) + self.K[:2, 2]  # 形状[M,2]
        boxes_img_corners3d = boxes_img_corners3d.astype(np.int32).reshape(-1, 8, 2)  # 转换为整数像素坐标      
        boxes_img_corners3d[:, :, 0] = np.clip(boxes_img_corners3d[:, :, 0], a_min=0, a_max=1919)
        boxes_img_corners3d[:, :, 1] = np.clip(boxes_img_corners3d[:, :, 1], a_min=0, a_max=1079)
        
        if  vis_img is not None:
            for i in range(boxes_img_corners3d.shape[0]):
                draw_projected_box3d(vis_img, boxes_img_corners3d[i, ...], text=str(det_3D_result[i, 7]))
            
            self.human_fusion_vis_pub.publish(self.bridge.cv2_to_imgmsg(vis_img, encoding="bgr8"))  
        
        # 根据投影角点计算人体中心像素横坐标，计算偏移量绝对值
        proj_center_x = np.mean(boxes_img_corners3d[:, :, 0], axis=1, keepdims=True)
        cur_face_center_x = np.array(cur_face_box)[:, 0].reshape(1, -1)
        pixel_dif = np.abs(proj_center_x - cur_face_center_x) # [n3d, n2d]

        
        # 初始化分配结果列表，-1 表示未分配
        assignments = [-1] * pixel_dif.shape[0]
        # 标记 2D 结果是否已被分配
        is_2d_assigned = [False] * pixel_dif.shape[1]

        # 按距离从小到大排序，获取索引
        sorted_indices = np.dstack(np.unravel_index(np.argsort(pixel_dif.ravel()), pixel_dif.shape))[0]

        for idx_3d, idx_2d in sorted_indices:
            if not is_2d_assigned[idx_2d] and assignments[idx_3d] == -1 and pixel_dif[idx_3d, idx_2d] <= self.pixel_thre:
                # 检查该 2D 结果是否距离此 3D 结果最近
                min_2d_distance = np.min(pixel_dif[:, idx_2d])
                if pixel_dif[idx_3d, idx_2d] == min_2d_distance:
                    assignments[idx_3d] = idx_2d
                    is_2d_assigned[idx_2d] = True
        
        for i in range(len(assignments)):
            cur_human = CustomHuman()
            cur_human.center_x = det_3D_result[i, 0]
            cur_human.center_y = det_3D_result[i, 1]
            cur_human.center_z = det_3D_result[i, 2]
            cur_human.length = det_3D_result[i, 3]
            cur_human.width = det_3D_result[i, 3]
            cur_human.height = det_3D_result[i, 5]
            cur_human.rotation = float(det_3D_result[i, 6])
            cur_human.track_id = int(det_3D_result[i, 7])
            
            if assignments[i] != -1:
                cur_face_kp = np.array(cur_face_kp).reshape(-1, 5, 2)
                cur_face_box = np.array(cur_face_box, dtype=int).reshape(-1, 4)
                print (assignments[i], cur_face_box, cur_face_kp, cur_face_name)

                cur_human.person_name = cur_face_name[assignments[i]]
                cur_human.face_x = int(cur_face_box[assignments[i], 0])
                cur_human.face_y = int(cur_face_box[assignments[i], 1])
                cur_human.face_size_x = int(cur_face_box[assignments[i], 2])
                cur_human.face_size_y = int(cur_face_box[assignments[i], 3])
                cur_human.lefteye_x = int(cur_face_kp[assignments[i], 0, 0])
                cur_human.lefteye_y = int(cur_face_kp[assignments[i], 0, 1])
                cur_human.righteye_x = int(cur_face_kp[assignments[i], 1, 0])
                cur_human.righteye_y = int(cur_face_kp[assignments[i], 1, 1])
                cur_human.nose_x = int(cur_face_kp[assignments[i], 2, 0])
                cur_human.nose_y = int(cur_face_kp[assignments[i], 2, 1])
                cur_human.leftlip_x = int(cur_face_kp[assignments[i], 3, 0])
                cur_human.leftlip_y = int(cur_face_kp[assignments[i], 3, 1])
                cur_human.rightlip_x = int(cur_face_kp[assignments[i], 4, 0])
                cur_human.rightlip_y = int(cur_face_kp[assignments[i], 4, 1])

            hpf_msg.human.append(cur_human)
        self.human_fusion_pub.publish(hpf_msg)
        self.get_logger().info("共看到{}个人和{}个人脸".format(hpf_msg.object_num, cur_face_num))

def main(args=None):
    # 用于人体相关的感知融合节点
    # 基于宇树D435i和mid360结果
    rclpy.init(args=args)
    node = Subscriber("HPF")
    rclpy.spin(node)
    rclpy.shutdown()
 
if __name__ == "__main__":
    main()