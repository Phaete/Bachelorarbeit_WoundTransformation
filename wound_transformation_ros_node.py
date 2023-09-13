# This script is part of the bachelor thesis
# "Localization of medical measurement data from injured persons for use in rescue robotics"
# from Konstantin Wenig written for the university to luebeck's bachelor degree in robotics and autonomous systems
# Redistribution and change without consent it not allowed

import cv2
import numpy as np
import yaml
import os
from matplotlib import pyplot as plt
import wound_transformation



try:
    import rospy
    from sensor_msgs.msg import Image, String
    from std_msgs.msg import Header
    import message_filters
except:
    print('Could not import rospy and related modules')
    exit(-1)


class WoundTransformROSNode:
    # def test(self):
    #     pass

    def __init__(self, **kwargs):
        self.base_path = os.path.dirname(
            os.path.abspath("postOpenPose.py")
        )

        # Create the proper wound transformation object based on the skeleton detector
        if 'skeleton' in kwargs and kwargs.get('skeleton') == 'nuitrack':
            self.wound_transform = wound_transformation.WoundTransformation(skeleton='nuitrack')
        else:
            self.wound_transform = wound_transformation.WoundTransformation()

        # Create ROS node for wound transformation
        rospy.init_node('wound_transformation_node', anonymous=True)

        self.converted_wounds_pub = rospy.Publisher('wound_transformation/out/converted_wounds', String, queue_size=3)
        self.standard_view_pub = rospy.Publisher('wound_transformation/out/standard_view', Image, queue_size=3)
        self.reprojection_error_pub = rospy.Publisher('wound_transformation/out/reprojection_error', String, queue_size=3)

        # Create message templates
        self.converted_wounds_msg = String()

        self.standard_view_msg = Image()
        self.standard_view_msg.width = 348
        self.standard_view_msg.height = 612

        self.reprojection_error_msg = String()

        # Create subscriber for wound detector output
        """
        The wounds are expected to come as an array of arrays following the example:
        [[x1, y1], [x2, y2], ..., [xn, yn]]
        for n wounds per image
        The wound detector is expected to attach the timestamp from the input images to the output to allow 
        synchronization
        """
        self.wound_sub = message_filters.Subsriber('wound_detector', String)

        # Create subscriber for skeleton detector output
        """
        The keypoints are expected to come as a dict following the example of
        self.keypoints = ...
        The rendered image is expected to be the rendered output image from openpose
        The skeleton detector is expected to attach the same timestamp as their input images to the keypoints and
        rendered image to allow synchronization
        """
        self.keypoints_sub = message_filters.Subscriber('skeleton_detector_keypoints', String)
        self.rendered_image_sub = message_filters.Subscriber('skeleton_detector_rendered', Image)
        self.depth_image_sub = message_filters.Subscriber('wound_transformation/realsense_d/image_rs', Image)

        # Synchronizer based on header timestamp
        self.time_sync = message_filters.TimeSynchronizer(
            [self.wound_sub, self.keypoints_sub, self.rendered_image_sub, self.depth_image_sub], 3
        )

        # Register callback
        self.time_sync.registerCallback(self.callback)

    def callback(self, wounds, keypoints, image, depth_image):
        # Step 1: Convert wounds from random pose to standard view reference frame
        _converted_wounds = self.wound_transform.locate_wounds(image, keypoints, wounds, depth_image, None)
        # Step 2: Transfer converted image onto the standard view
        _standard_view_converted_wounds = self.wound_transform.transform_view(_converted_wounds)
        # Step 3: Reproject converted wounds and calculate reprojection error
        _reprojected_wounds = self.wound_transform.reproject_wounds(keypoints, wounds, _converted_wounds)
        _error_per_wound = wound_transformation.calculate_reprojection_error(wounds, _reprojected_wounds)
        # Step 4: Publish all data on their respective Publishers
        # Grab current timestamp
        _timestamp = rospy.Time.now()

        # Create the header
        _header = Header()
        _header.stamp = _timestamp

        # Set message header and data and publish for converted wounds
        self.converted_wounds_msg.header = _header
        self.converted_wounds_msg.data = _converted_wounds.tolist()
        self.converted_wounds_pub.publish(self.converted_wounds_msg)

        # Set message header and data and publish for the standard view
        self.standard_view_msg.header = _header
        self.standard_view_msg.data = _standard_view_converted_wounds
        self.standard_view_pub.publish(self.standard_view_msg)

        # Set message header and data and publish for the reprojection error
        # Set message header and data and publish for the reprojection error
        self.reprojection_error_msg.header = _header
        self.reprojection_error_msg.data = _error_per_wound.tolist()
        self.reprojection_error_pub.publish(self.reprojection_error_msg)



