#!/usr/bin/python3
from __future__ import print_function
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from geometry_msgs.msg import PoseStamped
import struct
from sensor_msgs.msg import CameraInfo, Image
import rospy
import cv2
from cv_bridge import CvBridge
import numpy as np
import argparse
import random as rng
import tf
from tf import TransformListener
rng.seed(12345)
K = CameraInfo()
K_inv = [0]

pub = rospy.Publisher("point_cloud2", PointCloud2, queue_size=2)

fields = [PointField('x', 0, PointField.FLOAT32, 1),
          PointField('y', 4, PointField.FLOAT32, 1),
          PointField('z', 8, PointField.FLOAT32, 1),
          # PointField('rgb', 12, PointField.UINT32, 1),
          PointField('rgba', 12, PointField.UINT32, 1),
          ]

header = Header()
header.frame_id = "zed2i_right_camera_optical_frame"


def start_node():
    rospy.init_node('detect')
    rospy.Subscriber("/zed2i/zed_node/rgb/image_rect_color",
                     Image, process_image)
    rospy.spin()


def showImage(src):
    cv2.imshow("blobs",src)
    cv2.waitKey(1)

def process_image(msg):
    try:
        start_time = rospy.get_time()
        # convert sensor_msgs/Image to OpenCV Image
        bridge = CvBridge()
        orig = bridge.imgmsg_to_cv2(msg, "bgr8")
        src = orig
        hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

        # make sensitivity a param
        # sensitivity = rospy.get_param("/pothole/opencv-margin")
        sensitivity = rospy.get_param("/params/sensitivity")
        lower = np.array([0, 0, 255 - sensitivity])
        upper = np.array([255, sensitivity, 255])

        mask = cv2.inRange(hsv, lower, upper)

        res = cv2.bitwise_and(src, src, mask=mask)
        res = cv2.bitwise_not(res)

        # blob detection
        params = cv2.SimpleBlobDetector_Params()
        # Area filtering
        maxArea = rospy.get_param("/params/maxArea")
        minArea = rospy.get_param("/params/minArea")
        params.filterByArea = True
        params.minArea = minArea
        params.maxArea = maxArea

        # Circularity filtering
        cricularity = rospy.get_param("/params/cricularity")
        params.filterByCircularity = True
        params.minCircularity = cricularity

        convexity = rospy.get_param("/params/convexity")
        params.filterByConvexity = True
        params.minConvexity = convexity

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(res)

        # blank = np.ones((4, 4))
        margin = rospy.get_param("/params/margin")
        for i in keypoints:
            i.size += margin

        blobs = cv2.drawKeypoints(src, keypoints, 100, (255, 255, 255),
                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        pts = ([[key_point.pt, key_point.size] for key_point in keypoints])
        # print(pts)
        # sshowImage(blobs)
        point_holder = []
        rgb = struct.unpack('I', struct.pack('BBBB', 0, 255, 255, 100))[0]

        def call(msg):
            global K, K_inv
            K_inv = np.linalg.inv(np.array(msg.K).reshape([3, 3]))

        # Rotation matrix
        # Get yaw from tf
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

        rospy.Subscriber(
            "/zed2i/zed_node/rgb/camera_info", CameraInfo, call)

        # rospy.loginfo(K)

        def convert_point(vals):
            # vector = h * np.dot(K, vals) * h / \
            #     (np.dot(ground_normal_to_cam.T, np.dot(K, vals)))
            if len(K_inv) == 1:
                return np.zeros(3)
            uv_hom = np.array([vals[0], vals[1], 1])
            Kinv_uv = K_inv.dot(uv_hom) 
            denom = ground_normal_to_cam.dot(Kinv_uv)
            vector = h*Kinv_uv/denom
            # if rospy.get_param("test"):
            return vector

        points_per_pothole = 1000
        for i in pts:
            x = float(i[0][0])
            y = float(i[0][1])
            theta = 0
            increment = 2 * np.pi / points_per_pothole

            # vals = np.array([x + (size* np.cos(theta)), y + (size* np.sin(theta)), 0])
            vals = np.array([x, y, 0])
            vector = convert_point(vals)
            
            p1 = PoseStamped()
            p1.header.frame_id = "zed2i_right_camera_frame"
            p1.pose.position.x = vector[2] 
            p1.pose.position.y = -vector[0]
            p1.pose.position.z = -vector[1] 
            tf_listener = TransformListener()
            try:
                vector_base = tf_listener.transformPose("/base_link", p1)
                b_x = vector_base.pose.position.x
                b_y = vector_base.pose.position.y
                b_z = vector_base.pose.position.z
                
                for j in range(points_per_pothole):
                    theta += increment
                    point_holder += [[b_x + 0.35 *
                                      np.cos(theta), b_y + 0.35*np.sin(theta), b_z, rgb]]
            except:
                rospy.logwarn("tf unavailable")


        # rospy.loginfo(point_holder)

        pc2 = point_cloud2.create_cloud(header, fields, point_holder)
        pc2.header.frame_id = "base_link"
        pc2.header.stamp = rospy.Time.now()
        pub.publish(pc2)

        end_time = rospy.get_time()
        rospy.loginfo("IMAGE CALLBACK // TIME: %.2fms", (end_time -start_time)*1000)
        

        detector = cv2.SimpleBlobDetector_create(params)
        if rospy.get_param("/params/debug"):
            cv2.imshow("Filtering Circular Blobs Only", res)
            showImage(blobs)
        else:
            cv2.destroyAllWindows()

    except Exception as err:
        rospy.WARN(str(err))


if __name__ == '__main__':
    try:
        start_node()
    except rospy.ROSInterruptException:
        pass