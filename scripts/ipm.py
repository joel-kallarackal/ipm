#!/usr/bin/python3

import rclpy
from rclpy.node import Node

import rclpy.time
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo

from cv_bridge import CvBridge
import numpy as np
from sensor_msgs import point_cloud2

K = CameraInfo()
K_inv = [0]

class Transformer(Node):
    def __init__(self):
        super().__init__('ipm_node')
        
        # Publisher to publish the transformed points
        self.pc_publisher = self.create_publisher(PointCloud2, '/ipm', 10)
        
        # Subscribers
        self.img_rgb_sub = self.create_subscription(Image,'igvc/lanes_binary',self.get_binary,10)
        self.cam_info_sub = self.create_subscription(CameraInfo,"/zed2i/zed_node/rgb/camera_info", self.call)
        
    def call(self, data):
        global K, K_inv
        K_inv = np.linalg.inv(np.array(data.K).reshape([3, 3]))
        
    
    def get_binary(self, data):
        binary_image = CvBridge().imgmsg_to_cv2(data,desired_encoding="passthrough")
        world_coordinates = self.get_world_coordinates(binary_image)
        
        pc = self.point_cloud(world_coordinates,"zed_camera_left_optical_frame")
        
        self.pc_publisher.publish(pc)
        
        # Make Point cloud and publish it
        
    def get_world_coordinates(self,binary_image):
        binary_array = binary_image
        prob_left = binary_array
        prob_left = prob_left / 255.
        binary_coordinates = np.column_stack(np.where(binary_array == 255))
        
        points = np.array([])
        
        for point in binary_coordinates:
            world_coord = self.convert_point(point)
            points = np.append(points,np.array(world_coord))
           
        return points
        
        
    
    def convert_point(self, coords):
        roll = 0
        pitch = -0.4363
        yaw = 0
        h = 0.96
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)
        rotation_ground_to_cam = np.array([[cr*cy+sp*sr+sy, cr*sp*sy-cy*sr, -cp*sy],
                                           [cp*sr, cp*cr, sp],
                                           [cr*sy-cy*sp*sr, -cr*cy*sp - sr*sy, cp*cy]])

        # inv of rotation mat is same as its transpose
        rotation_cam_to_ground = rotation_ground_to_cam.T

        n = np.array([0, 1, 0])
        ground_normal_to_cam = (rotation_cam_to_ground.T).dot(n)  # nc
        
        if len(K_inv) == 1:
                return np.zeros(3)
        uv_hom = np.array([coords[0], coords[1], 1])
        Kinv_uv = K_inv.dot(uv_hom) 
        denom = ground_normal_to_cam.dot(Kinv_uv)
        vector = h*Kinv_uv/denom

        return vector
    
    def point_cloud(self,points, parent_frame):
        """ Creates a point cloud message.
        Args:
            points: Nx3 array of xyz positions.
            parent_frame: frame in which the point cloud is defined
        Returns:
            sensor_msgs/PointCloud2 message
        """
        ros_dtype = PointField.FLOAT32
        itemsize = np.dtype(np.float32).itemsize  # A 32-bit float takes 4 bytes.
        # data = points.astype(dtype).tobytes()
        data = np.array(points,dtype=np.float32).tobytes()
        fields = [PointField(name=n, offset=i * itemsize, datatype=ros_dtype, count=1) for i, n in enumerate('xyz')]
        header = Header()
        header.frame_id=parent_frame

        return PointCloud2(
            header=header,
            height=1,
            width=points.shape[0],
            is_dense=False,
            is_bigendian=False,
            fields=fields,
            point_step=(itemsize * 3),  # Every point consists of three float32s.
            row_step=(itemsize * 3 * points.shape[0]),
            data=data
        )

    
def main(args=None):
    rclpy.init(args=args)
    transformer = Transformer(Node)
    rclpy.spin(transformer)
    transformer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()