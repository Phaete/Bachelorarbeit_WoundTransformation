# This script is part of the bachelor thesis
# "Localization of medical measurement data from injured persons for use in rescue robotics"
# from Konstantin Wenig written for the university to luebeck's bachelor degree in robotics and autonomous systems
# Redistribution and change without consent it not allowed

import cv2
import numpy as np
import yaml
import thermal_cam_calibration
import realsense_calibration
import stereoCalibration
import os
try:
    import rospy
    from sensor_msgs.msg import Image, CameraInfo
    from std_msgs.msg import Header
    from cv_bridge import CvBridge, CvBridgeError
except:
    print('Could not import rospy and messages')


def convert_16_1_to_8_2(image_16_bit):
    upper_8_bits = np.uint8(image_16_bit >> 8)
    lower_8_bits = np.uint8(image_16_bit & 0xFF)
    return np.dstack((upper_8_bits, lower_8_bits))


def main(num_thermal, num_rs, num_stereo, **kwargs):
    _base_path = os.path.dirname(
        os.path.abspath("preOpenPose.py")
    )
    print('Checking for directories...')
    # Create directory for thermal calibration pictures if it does not exist
    path_thermal_calib = _base_path + "/assets/pictures_thermal_calib/"
    if not os.path.exists(path_thermal_calib):
        os.makedirs(path_thermal_calib)
        print('Directory for thermal calibration pictures created.')
    # Create directory for realsense calibration pictures if it does not exist
    path_rs_calib = _base_path + "/assets/pictures_rs_calib/"
    if not os.path.exists(path_rs_calib):
        os.makedirs(path_rs_calib)
        print('Directory for realsense calibration pictures created.')
    # Create directory for stereo calibration pictures if it does not exist
    path_stereo_calib = _base_path + "/assets/pictures_stereo_calib/"
    if not os.path.exists(path_stereo_calib):
        os.makedirs(path_stereo_calib)
        print('Directory for stereo calibration pictures created.')
    # Create directory for the realsense color pictures if it does not exist
    path_img_rs_color = _base_path + "/assets/pictures/rs_color/"
    if not os.path.exists(path_img_rs_color):
        os.makedirs(path_img_rs_color)
        print('Directory for rs_color pictures created.')
    # Create directory for the realsense depth pictures if it does not exist
    path_img_rs_depth = _base_path + "/assets/pictures/rs_depth/"
    if not os.path.exists(path_img_rs_depth):
        os.makedirs(path_img_rs_depth)
        print('Directory for rs_depth pictures created.')
    # Create directory for the thermal pictures if it does not exist
    path_img_thermal = _base_path + "/assets/pictures/thermal/"
    if not os.path.exists(path_img_thermal):
        os.makedirs(path_img_thermal)
        print('Directory for thermal pictures created.')
    # Create directory for the wounds if it does not exist
    path_wounds = _base_path + "/assets/pictures/wounds/"
    if not os.path.exists(path_wounds):
        os.makedirs(path_wounds)
        print('Directory for wounds created.')
    # Create directory for the calibration files if it does not exist
    path_calib = _base_path + "/assets/calibration/"
    if not os.path.exists(path_calib):
        os.makedirs(path_calib)
        print('Directory for calibration files created.')
    # Create directory for the openpose keypoints if it does not exist
    path_keypoints = _base_path + "/assets/pictures/keypoints/"
    if not os.path.exists(path_keypoints):
        os.makedirs(path_keypoints)
        print('Directory for calibration files created.')

    # Check if any previous realsense calibration has been done and should be used
    path_rs_calib_file = path_rs_calib + "calibration_rs.yaml"
    print("Checking for previous realsense calibration...")
    if os.path.exists(path_rs_calib_file):
        str_input = input("Previous calibration found. Type \"y\" if you want to re-use the calibration or "
                          "type \"n\" if you want to re-calibrate.")
        if str_input == 'y':
            # Read config from file
            with open(path_rs_calib_file, 'r') as stream:
                dictionary = yaml.safe_load(stream)
                rs_calib = realsense_calibration.RealsenseCalibration(mode=1,
                                                                      mtx=np.float64(dictionary.get("camera_matrix")),
                                                                      dist=np.float64(dictionary.get("dist_coeff")),
                                                                      rvecs=np.float64(dictionary.get("rvecs")),
                                                                      tvecs=np.float64(dictionary.get("tvecs")))
        else:
            count = len([f for f in os.listdir(path_rs_calib) if f.startswith("rs_grid_")])
            rs_calib = realsense_calibration.RealsenseCalibration(
                mode=0, num=num_rs, found=0, path=path_rs_calib, cam=(count < num_rs)
            )
            # Count proper calibration images, by counting the images that have already found the circle grid
            # and only take new images, if the number of images is not the same as num_rs
            if count < num_rs:
                rs_calib.take_pictures()
            print("Recalibrating ...")
            rs_calib.calibrate()
    else:
        print("No previous calibration found.")
        count = len([f for f in os.listdir(path_rs_calib) if f.startswith("rs_grid_")])
        rs_calib = realsense_calibration.RealsenseCalibration(
            mode=0, num=num_rs, found=0, path=path_rs_calib, cam=(count < num_rs)
        )
        # Count proper calibration images, by counting the images that have already found the circle grid
        # and only take new images, if the number of images is not the same as num_rs
        if count < num_rs:
            rs_calib.take_pictures()
        rs_calib.calibrate()

    # Check if any previous thermal calibration has been done and should be used
    path_thermal_calib_file = path_thermal_calib + "calibration_thermal.yaml"
    print("Checking for previous thermal calibration...")
    if os.path.exists(path_thermal_calib_file):
        str_input = input("Previous calibration found. Type \"y\" if you want to re-use the calibration or "
                          "type \"n\" if you want to re-calibrate.")
        if str_input == 'y':
            # Read config from file
            with open(path_thermal_calib_file, 'r') as stream:
                dictionary = yaml.safe_load(stream)
                thermal_calib = thermal_cam_calibration.ThermalCalibration(mode=1,
                                                                           mtx=np.float64(
                                                                               dictionary.get("camera_matrix")),
                                                                           dist=np.float64(
                                                                               dictionary.get("dist_coeff")),
                                                                           rvecs=np.float64(dictionary.get("rvecs")),
                                                                           tvecs=np.float64(dictionary.get("tvecs"))
                                                                           )
        else:
            count = len([f for f in os.listdir(path_thermal_calib) if f.startswith("thermal_")])
            thermal_calib = thermal_cam_calibration.ThermalCalibration(
                mode=0, num=num_thermal, found=0, path=path_thermal_calib, cam=(count < num_rs)
            )
            # Count proper calibration images, by counting the images that have already found the circle grid
            # and only take new images, if the number of images is not the same as num_thermal
            if count < num_thermal:
                thermal_calib.take_pictures(scale_fac=8)
            print("Recalibrating ...")
            thermal_calib.calibrate()
    else:
        count = len([f for f in os.listdir(path_thermal_calib) if f.startswith("thermal_")])
        thermal_calib = thermal_cam_calibration.ThermalCalibration(
            mode=0, num=num_thermal, found=0, path=path_thermal_calib, cam=(count < num_rs)
        )
        # Count proper calibration images, by counting the images that have already found the circle grid
        # and only take new images, if the number of images is not the same as num_thermal
        if count < num_thermal:
            thermal_calib.take_pictures(scale_fac=8)
        thermal_calib.calibrate()

    # Check if any previous calibration has been done and should be used
    path_stereo_calib_file = path_stereo_calib + "calibration_stereo.yaml"
    print("Checking for previous stereo calibration...")
    if os.path.exists(path_stereo_calib_file):
        str_input = input("Previous calibration found. Type \"y\" if you want to re-use the calibration or "
                          "type \"n\" if you want to re-calibrate.")
        if str_input == 'y':
            # Read config from file
            with open(path_stereo_calib_file, 'r') as stream:
                dictionary = yaml.safe_load(stream)
                stereo_calib = stereoCalibration.StereoCalibration(mode=1,
                                                                   cmtx_rs=np.float64(
                                                                       dictionary.get('camera_matrix_rs')),
                                                                   dist_rs=np.float64(
                                                                       dictionary.get('dist_coeff_rs')),
                                                                   cmtx_thermal=np.float64(
                                                                       dictionary.get('camera_matrix_thermal')),
                                                                   dist_thermal=np.float64(
                                                                       dictionary.get('dist_coeff_thermal')),
                                                                   rmtx=np.float64(
                                                                       dictionary.get('R_matrix')),
                                                                   tmtx=np.float64(
                                                                       dictionary.get('t_matrix')),
                                                                   essential_mtx=np.float64(
                                                                       dictionary.get('essential_matrix')),
                                                                   fundamental_mtx=np.float64(
                                                                       dictionary.get('fundamental_matrix')),
                                                                   per_view_errors=np.float64(
                                                                       dictionary.get('reprojection_error'))
                                                                   )
        else:
            count_thermal = len([f for f in os.listdir(path_stereo_calib) if f.startswith("thermal_grid_")])
            count_rs = len([f for f in os.listdir(path_stereo_calib) if f.startswith("rs_grid_")])
            cams = (count_rs < num_stereo) and (count_thermal < num_stereo)
            stereo_calib = stereoCalibration.StereoCalibration(
                mode=0, num=num_stereo, found=0, path=path_stereo_calib, cam=cams
            )
            # Count proper calibration images, by counting the images that have already found the circle grid
            # and only take new images, if the number of images is not the same as num_stereo
            if count_rs < num_stereo or count_thermal < num_stereo:
                stereo_calib.take_pictures(8)
            print("Recalibrating ...")
            stereo_calib.calibrate(rs_calib.mtx, thermal_calib.mtx, rs_calib.dist, thermal_calib.dist)
            print(stereo_calib.cmtx_rs.dtype)
    else:
        count_thermal = len([f for f in os.listdir(path_stereo_calib) if f.startswith("thermal_grid_")])
        count_rs = len([f for f in os.listdir(path_stereo_calib) if f.startswith("rs_grid_")])
        cams = (count_rs < num_stereo) and (count_thermal < num_stereo)
        stereo_calib = stereoCalibration.StereoCalibration(
            mode=0, num=num_stereo, found=0, path=path_stereo_calib, cam=cams
        )
        # Count proper calibration images, by counting the images that have already found the circle grid
        # and only take new images, if the number of images is not the same as num_stereo
        if count_rs < num_stereo or count_thermal < num_stereo:
            stereo_calib.take_pictures(8)
        stereo_calib.calibrate(rs_calib.mtx, thermal_calib.mtx, rs_calib.dist, thermal_calib.dist)
"""
    # After the calibration is done, we can create some images to be used in other modules (openpose/wound detection)
    i = 1
    # Reload thermal and rs calibration as previous calibration has either not started those or closed them at the end
    with open(path_rs_calib_file, 'r') as stream:
        dictionary = yaml.safe_load(stream)
        rs_calib = realsense_calibration.RealsenseCalibration(mode=2,
                                                              mtx=np.float64(dictionary.get("camera_matrix")),
                                                              dist=np.float64(dictionary.get("dist_coeff")),
                                                              rvecs=np.float64(dictionary.get("rvecs")),
                                                              tvecs=np.float64(dictionary.get("tvecs")))
    with open(path_thermal_calib_file, 'r') as stream:
        dictionary = yaml.safe_load(stream)
        thermal_calib = thermal_cam_calibration.ThermalCalibration(mode=2,
                                                                   mtx=np.float64(dictionary.get("camera_matrix")),
                                                                   dist=np.float64(dictionary.get("dist_coeff")),
                                                                   rvecs=np.float64(dictionary.get("rvecs")),
                                                                   tvecs=np.float64(dictionary.get("tvecs")))
    # Make some images for OpenPose / the wound detection to analyze
    if 'host' in kwargs and kwargs.get('host') == 'local':
        while True:
            # Grab frames first, so they are mostly synched
            # rs_color, rs_depth = rs_calib.rs.get_frame()
            rs_rgb, rs_d = rs_calib.rs.get_frame()
            thermal_img = thermal_calib.thermal.get_frame()
            thermal_img_big = cv2.resize(thermal_img, (0, 0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
            # rs_depth_uint8 = cv2.normalize(rs_d, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            # Undistort rs images
            # h_rs, w_rs = rs_rgb.shape[:2]
            # rs_cam_mtx, roi_rs = cv2.getOptimalNewCameraMatrix(stereo_calib.cmtx_rs,
            #                                                   stereo_calib.dist_rs,
            #                                                   (w_rs, h_rs), 0, (w_rs, h_rs))
            # rs_color_undistorted = cv2.undistort(rs_rgb,
            #                                     stereo_calib.cmtx_rs, stereo_calib.dist_rs,
            #                                     None, rs_cam_mtx)
            # rs_depth_undistorted = cv2.undistort(rs_d,
            #                                     stereo_calib.cmtx_rs, stereo_calib.dist_rs,
            #                                     None, rs_cam_mtx)
            # rs_color_undistorted = cv2.undistort(rs_rgb,
            #                                      stereo_calib.cmtx_rs, stereo_calib.dist_rs,
            #                                      None, stereo_calib.cmtx_rs)
            # rs_depth_undistorted = cv2.undistort(rs_d,
            #                                      stereo_calib.cmtx_rs, stereo_calib.dist_rs,
            #                                      None, stereo_calib.cmtx_rs)

            # Undistort thermal images
            # try:
            #    h_th, w_th, _ = thermal_img.shape
            # except:
            #     continue
            # thermal_cam_mtx, roi_th = cv2.getOptimalNewCameraMatrix(stereo_calib.cmtx_thermal, stereo_calib.dist_thermal, (w_th, h_th), 0,
            #                                                        (w_th, h_th))
            # Undistort with OptimalNewCameraMatrix or not?
            # thermal_cam_mtx, roi_th = cv2.getOptimalNewCameraMatrix(thermal_calib.mtx, thermal_calib.dist, (w_th, h_th), 0,
            #                                                        (w_th, h_th))


            cv2.imshow('rs_dist', rs_rgb)
            # cv2.imshow('rs_undist', rs_color_undistorted)
            # cv2.imshow('rs_depth', rs_depth_uint8)
            cv2.imshow('rs_depth', rs_d)
            cv2.imshow('th_dist', thermal_img_big)
            # cv2.imshow('th_undist', thermal_img_undistorted)
            # cv2.imshow("image", img)
            key = cv2.waitKey(10)
            if key == 32:  # Press space to save image
                # Save images
                _filename_rs_color = path_img_rs_color + str(i) + ".jpg"
                # cv2.imwrite(_filename_rs_color, rs_color_undistorted)
                cv2.imwrite(_filename_rs_color, rs_rgb)

                _filename_rs_depth = path_img_rs_depth + str(i) + ".jpg"
                # cv2.imwrite(_filename_rs_depth, rs_depth_undistorted)
                cv2.imwrite(_filename_rs_depth, rs_d)

                _filename_thermal = path_img_thermal + str(i) + ".jpg"
                cv2.imwrite(_filename_thermal, thermal_img_big)
                i += 1
            elif key == 33:  # Press ESC to stop the loop
                rs_calib.rs.stop_pipeline
                thermal_calib.thermal.stop_stream
                break
    else:
        # Online via ROS node

        # Images and camera info will be published according to:
        # http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/CameraInfo.html
        # > It should be in a camera namespace on topic "camera_info" and accompanied by up to five image topics named:
        # > image_raw        - raw data from the camera driver, possibly Bayer encoded
        # > image            - monochrome, distorted
        # > image_color      - color, distorted
        # > image_rect       - monochrome, rectified
        # > image_rect_color - color, rectified

        # Create node
        rospy.init_node('wound_transform_image_publisher')

        _bridge = CvBridge()

        _fps = 15
        _rate = rospy.Rate(_fps)

        # Publishers for camera info and images from the realsense d455 depth camera
        _info_rs_pub = rospy.Publisher('wound_transformation/realsense_rgb/camera_info_rs',
                                       CameraInfo, queue_size=1)
        # Raw image, directly from rgb camera
        _image_rs_rgb_raw_pub = rospy.Publisher('wound_transformation/realsense_rgb/image_rs_raw',
                                                Image, queue_size=1)
        # Distorted image in format used for calibration, grayscale
        _image_rs_rgb_pub = rospy.Publisher('wound_transformation/realsense_rgb/image_rs',
                                            Image, queue_size=1)
        # Distorted image in format used for calibration, color
        _image_rs_rgb_color_pub = rospy.Publisher('wound_transformation/realsense_rgb/image_rs_color',
                                                  Image, queue_size=1)
        # Rectified image in format used for calibration, grayscale
        _image_rs_rgb_rect_pub = rospy.Publisher('wound_transformation/realsense_rgb/image_rs_rect',
                                                 Image, queue_size=1)
        # Rectified image in format used for calibration, grayscale
        _image_rs_rgb_rect_color_pub = rospy.Publisher('wound_transformation/realsense_rgb/image_rs_rect_color',
                                                       Image, queue_size=1)

        # Raw image, directly from depth camera
        _image_rs_d_raw_pub = rospy.Publisher('wound_transformation/realsense_d/image_rs_raw',
                                              Image, queue_size=1)
        # Distorted image in format used for calibration, grayscale
        _image_rs_d_pub = rospy.Publisher('wound_transformation/realsense_d/image_rs',
                                          Image, queue_size=1)
        # Rectified image in format used for calibration, grayscale
        _image_rs_d_rect_pub = rospy.Publisher('wound_transformation/realsense_d/image_rs_rect',
                                               Image, queue_size=1)

        # ---------

        # Publishers for camera info and images from the PureThermal Mini Pro JST-SR thermal imaging camera
        _info_thermal_pub = rospy.Publisher('wound_transformation/pure_thermal/camera_info_thermal',
                                            CameraInfo, queue_size=1)
        # Raw image, directly from thermal camera
        _image_thermal_raw_pub = rospy.Publisher('wound_transformation/pure_thermal/image_thermal_raw',
                                                 Image, queue_size=1)
        # Distorted image in format used for calibration, grayscale
        _image_thermal_pub = rospy.Publisher('wound_transformation/pure_thermal/image_thermal',
                                             Image, queue_size=1)
        # Rectified image in format used for calibration, grayscale
        _image_thermal_rect_pub = rospy.Publisher('wound_transformation/pure_thermal/image_thermal_rect',
                                                  Image, queue_size=1)

        # Stereo rectification
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            stereo_calib.cmtx_rs,
            stereo_calib.dist_rs,
            stereo_calib.cmtx_thermal,
            stereo_calib.dist_thermal,
            (480, 480),
            stereo_calib.rmtx,
            stereo_calib.tmtx,
            alpha=0.98
        )
        # Create maps to be used for remapping to create rectified images
        rmap_rs_x, rmap_rs_y = cv2.initUndistortRectifyMap(
            stereo_calib.cmtx_rs,
            stereo_calib.dist_rs,
            R1,
            P1,
            (480, 480),
            cv2.CV_8UC1  # cv2.CV_16SC2
        )
        rmap_thermal_x, rmap_thermal_y = cv2.initUndistortRectifyMap(
            stereo_calib.cmtx_thermal,
            stereo_calib.dist_thermal,
            R2,  # R2,
            P2,
            (480, 480),
            cv2.CV_8UC1  # cv2.CV_16SC2
        )

        # Create CameraInfo message
        # _info_rs
        _info_rs_msg = CameraInfo()
        _info_rs_msg.width = 480
        _info_rs_msg.height = 480
        _info_rs_msg.K = stereo_calib.cmtx_rs
        _info_rs_msg.D = stereo_calib.dist_rs
        _info_rs_msg.R = R1
        _info_rs_msg.P = P1
        _info_rs_msg.distortion_model = 'plumb_bob'

        # _info_thermal
        _info_thermal_msg = CameraInfo()
        _info_thermal_msg.width = 480
        _info_thermal_msg.height = 480
        _info_thermal_msg.K = stereo_calib.cmtx_thermal
        _info_thermal_msg.D = stereo_calib.dist_thermal
        _info_thermal_msg.R = R2
        _info_thermal_msg.P = P2
        _info_thermal_msg.distortion_model = 'plumb_bob'

        # Create templates for image messages
        # rs rgb
        # rs_rgb_raw
        _image_rs_rgb_raw_msg = Image()
        _image_rs_rgb_raw_msg.width = 640
        _image_rs_rgb_raw_msg.height = 480
        # rs_rgb
        _image_rs_rgb_msg = Image()
        _image_rs_rgb_msg.width = 480
        _image_rs_rgb_msg.height = 480
        # rs_rgb_color
        _image_rs_rgb_color_msg = Image()
        _image_rs_rgb_color_msg.width = 480
        _image_rs_rgb_color_msg.height = 480
        # rs_rgb_rect
        _image_rs_rgb_rect_msg = Image()
        _image_rs_rgb_rect_msg.width = 480
        _image_rs_rgb_rect_msg.height = 480
        # rs_rgb_rect_color
        _image_rs_rgb_rect_color_msg = Image()
        _image_rs_rgb_rect_color_msg.width = 480
        _image_rs_rgb_rect_color_msg.height = 480

        # rs depth
        # rs_d_raw
        _image_rs_d_raw_msg = Image()
        _image_rs_d_raw_msg.width = 640
        _image_rs_d_raw_msg.height = 480
        # rs_d
        _image_rs_d_msg = Image()
        _image_rs_d_msg.width = 480
        _image_rs_d_msg.height = 480
        # rs_d_rect
        _image_rs_d_rect_msg = Image()
        _image_rs_d_rect_msg.width = 480
        _image_rs_d_rect_msg.height = 480

        # thermal
        # thermal_raw
        _image_thermal_raw_msg = Image()
        _image_thermal_raw_msg.width = 60
        _image_thermal_raw_msg.height = 60
        # thermal
        _image_thermal_msg = Image()
        _image_thermal_msg.width = 480
        _image_thermal_msg.height = 480
        # thermal_rect
        _image_thermal_rect_msg = Image()
        _image_thermal_rect_msg.width = 480
        _image_thermal_rect_msg.height = 480

        while not rospy.is_shutdown():

            # Grab raw rs images
            _image_rs_rgb_raw, _image_rs_d_raw = rs_calib.rs.get_frame()  # _image_rs_raw[0] and _image_rs_raw[1] are the raw rgb/d images
            # Grab raw thermal image
            _image_thermal_raw = thermal_calib.thermal.get_frame()  # raw thermal image
            
            # Distorted images in format of calibration
            # rs_rgb (color and grayscale)
            _image_rs_rgb_color = _image_rs_rgb_raw[:, 80:560]
            _image_rs_rgb = cv2.cvtColor(_image_rs_rgb_color, cv2.COLOR_BGR2GRAY)
            # rs_d (grayscale only)
            _image_rs_d = _image_rs_d_raw[:, 80:560]
            # thermal (grayscale only)
            _image_thermal = stereoCalibration.raw_to_8bit(_image_thermal_raw)
            _image_thermal = cv2.resize(_image_thermal, (0, 0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)

            # Rectified images
            # rs_rgb_rect (color and grayscale)
            _image_rs_rgb_rect_color = cv2.remap(_image_rs_rgb_color, rmap_rs_x, rmap_rs_y, interpolation=cv2.INTER_CUBIC)
            _image_rs_rgb_rect = cv2.remap(_image_rs_rgb, rmap_rs_x, rmap_rs_y, interpolation=cv2.INTER_CUBIC)
            # rs_d_rect (grayscale only)
            _image_rs_d_rect = cv2.remap(_image_rs_d, rmap_rs_x, rmap_rs_y, interpolation=cv2.INTER_CUBIC)
            # thermal_rect (grayscale only)
            _image_thermal_rect = cv2.remap(_image_thermal, rmap_thermal_x, rmap_thermal_y, interpolation=cv2.INTER_CUBIC)

            # ---------

            # Grab current timestamp
            _timestamp = rospy.Time.now()

            # Create the header
            _header = Header()
            _header.stamp = _timestamp

            # Add current timestamp to header
            # _info_rs
            _info_rs_msg.header = _header

            # _info_thermal
            _info_thermal_msg.header = _header

            # Add current timestamp and data to image messages
            # rs rgb
            # rs_rgb_raw
            _image_rs_rgb_raw_msg.header = _header
            _image_rs_rgb_raw_msg.data = _image_rs_rgb_raw
            # rs_rgb
            _image_rs_rgb_msg.header = _header
            _image_rs_rgb_msg.data = _image_rs_rgb
            # rs_rgb_color
            _image_rs_rgb_color_msg.header = _header
            _image_rs_rgb_color_msg.data = _image_rs_rgb_color
            # rs_rgb_rect
            _image_rs_rgb_rect_msg.header = _header
            _image_rs_rgb_rect_msg.data = _image_rs_rgb_rect
            # rs_rgb_rect_color
            _image_rs_rgb_rect_color_msg.header = _header
            _image_rs_rgb_rect_color_msg.data = _image_rs_rgb_rect_color

            # rs depth
            # rs_d_raw
            _image_rs_d_raw_msg.header = _header
            _image_rs_d_raw_msg.data = _image_rs_d_raw
            # rs_d
            _image_rs_d_msg.header = _header
            _image_rs_d_msg.data = _image_rs_d
            # rs_d_rect
            _image_rs_d_rect_msg.header = _header
            _image_rs_d_rect_msg.data = _image_rs_d_rect

            # thermal
            # thermal_raw
            _image_thermal_raw_msg.header = _header
            _image_thermal_raw_msg.data = _image_thermal_raw
            # thermal
            _image_thermal_msg.header = _header
            _image_thermal_msg.data = _image_thermal
            # thermal_rect
            _image_thermal_rect_msg.header = _header
            _image_thermal_rect_msg.data = _image_thermal_rect

            # Publish messages
            _info_rs_pub.publish(_info_rs_msg)
            _image_rs_rgb_raw_pub.publish(_image_rs_rgb_raw_msg)
            _image_rs_rgb_pub.publish(_image_rs_rgb_msg)
            _image_rs_rgb_color_pub.publish(_image_rs_rgb_color_msg)
            _image_rs_rgb_rect_pub.publish(_image_rs_rgb_rect_msg)
            _image_rs_rgb_rect_color_pub.publish(_image_rs_rgb_rect_color_msg)
            _image_rs_d_raw_pub.publish(_image_rs_d_raw_msg)
            _image_rs_d_pub(_image_rs_d_msg)
            _image_rs_d_rect_pub(_image_rs_d_rect_msg)

            _info_thermal_pub.publish(_info_thermal_msg)
            _image_thermal_raw_pub.publish(_image_thermal_raw_msg)
            _image_thermal_pub.publish(_image_thermal_msg)
            _image_thermal_rect_pub.publish(_image_thermal_rect_msg)

            # Pause thread
            _rate.sleep()
"""

if __name__ == "__main__":
    main(10, 10, 10, host='local')


