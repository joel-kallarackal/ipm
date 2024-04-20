#!/usr/bin/python3

import numpy as np
from custom_msgs.msg import ArrayXY
from camera_geometry import CameraGeometry
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
from numpy import unique
from numpy import where
from sklearn.cluster import DBSCAN


counter = 1
class LaneDetector(Node):
    def __init__(self):

        super().__init__('lane_detector')
        self.img_rgb_sub = self.create_subscription(
            Image,
            'igvc/lanes_binary',
            self.get_binary,
            10)
        self.img_rgb_sub
        
        self.lane_xy_pub = self.create_publisher(ArrayXY, 'igvc/lanes_xy_array', 10)
        self.img_curve_pub = self.create_publisher(Float32MultiArray, 'igvc/lane_coeffs', 1)
        self.lane_waypoint = self.create_publisher(Float32MultiArray, 'igvc/waypoint_gen', 1)

    def lane_waypointpublisher(self,coeff):
        if len(coeff) == 3:

            x_curve = np.linspace(1,5, 100)
            # Compute the corresponding y values using the polynomial coefficients
            ycurve_left = np.polyval(coeff[2], x_curve)
            ycurve_middle = np.polyval(coeff[1],x_curve)
            distances = ycurve_left - ycurve_middle
            variance = np.var(distances)
            if variance < 0.1:
                print(variance)
                x_value = 5  # The x-value at which you want to calculate the midpoint

                # Evaluate the first line at x = 5
                y1 = np.polyval(coeff[2], x_value)

                # Evaluate the second line at x = 5
                y2 = np.polyval(coeff[1], x_value)

                # Calculate the midpoint of y-values
                y_mid = (y1 + y2) / 2
                msg = Float32MultiArray()
                msg.data = np.array([x_value,y_mid])
                self.lane_waypoint.publish(msg)
        


    def get_binary(self, data):
        binary_image = CvBridge().imgmsg_to_cv2(data,desired_encoding="passthrough")
        world_coordinates = self.get_world_coordinates(binary_image)
        # self.publish_lane_coefficients(world_coordinates[2])
        self.publish_xy(world_coordinates[0],world_coordinates[1])
        # self.lane_waypointpublisher(world_coordinates[2])

    def publish_lane_coefficients(self,coeffs):
        # Create Float32MultiArray messages for each publisher
        lane_coeff_msg = Float32MultiArray()
        # Populate the lane coefficient data for each message
        lane_coeff_msg.data = list(np.array(coeffs,dtype="float64").flatten())

        # Publish the messages
        self.img_curve_pub.publish(lane_coeff_msg)

    def get_world_coordinates(self,binary_image):
        # binary_array = np.asarray(binary_image)
        binary_array = binary_image
        prob_left = binary_array
        prob_left = prob_left / 255.
        cg = CameraGeometry(image_height=binary_image.shape[0],image_width=binary_image.shape[1])
        binary_coordinates = np.column_stack(np.where(binary_array == 255))
        print("hello")
        xyz = cg.uv_coordinates_to_roadXYZ_roadframe_iso8855(binary_coordinates[:,[1,0]])
        x = xyz[:,0]
        y = xyz[:,1]
        # Clustering
        # clustered_output = self.clustering(x,y)
        # clustered_x=[]
        # clustered_y=[]
        # # Your original list of curves
        # print(np.shape(clustered_output))
        # for i in range(len(clustered_output)):
        #     clustered_x=np.concatenate((clustered_x,clustered_output[0]))
        #     clustered_y=np.concatenate((clustered_y,clustered_output[1]))
        return (x,y)

    def publish_xy(self,x_arr,y_arr):
        msg = ArrayXY()
        msg.x = list(x_arr)
        msg.y = list(y_arr)
        self.lane_xy_pub.publish(msg)
        
        global counter
        # print(f"{counter} frames have been published by now!!")
        counter = counter+1
        
    def clustering(self,x,y):
        X = np.column_stack((x,y))
        mask = np.logical_and(x > -1, x < 15)
        X = X[mask]
        downsampling_factor = 10
        # Downsample the points
        X = X[::downsampling_factor]
        curves = []
        lane_clusters = []
        return X


def main(args=None):
    rclpy.init(args=args)
    lane_detector = LaneDetector()
    rclpy.spin(lane_detector)
    lane_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
