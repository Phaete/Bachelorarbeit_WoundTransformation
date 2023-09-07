# This script is part of the bachelor thesis
# "Localization of medical measurement data from injured persons for use in rescue robotics"
# from Konstantin Wenig written for the university to luebeck's bachelor degree in robotics and autonomous systems
# Redistribution and change without consent it not allowed

import cv2
import numpy as np
import yaml
import thermal_cam
import realsense_cam
import traceback
import random
import os
from matplotlib import pyplot as plt


# Convert 16-bit to 8-bit
def raw_to_8bit(data):
    _data_norm = cv2.normalize(data, None, 0, 256, cv2.NORM_MINMAX)
    return cv2.cvtColor(np.uint8(_data_norm), cv2.COLOR_GRAY2RGB)


class StereoCalibration:
    def __init__(self, **kwargs):
        if kwargs.get('mode') == 0:  # mode = 0 -> no calibration found or being repeated
            # Variables for the camera calibration
            # Including variables for the calibration object
            self.pattern_rows = 4
            self.pattern_columns = 11
            self.distance_world_unit = 1
            # And arrays to store object points and image points from all the images
            self.objpoints = []  # 3d point in real world space
            self.imgpoints_rs = []  # 2d points in image plane
            self.imgpoints_thermal = []  # 2d points in image plane
            # Flags for stereo calibration
            self.stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
            # Termination criteria for the calibration
            self.termin_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
            # Preparation for the object points
            self.objp = np.array(
                [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [0.5, 0.5, 0], [1.5, 0.5, 0], [2.5, 0.5, 0], [3.5, 0.5, 0]], np.float32)
            for y in range(2, 11):
                for x in range(4):
                    self.objp = np.append(self.objp,
                                          [np.array([self.objp[4 * (y - 2) + x][0], self.objp[4 * (y - 2) + x][1] + 1, 0], np.float32)],
                                          axis=0)
            # Variables for the amount of pictures that should be taken and processed before starting calibration
            self.num = kwargs.get('num')
            self.found = kwargs.get('found')
            if 'cam' in kwargs and kwargs.get('cam') == 1:
                # Create camera object
                self.rs = realsense_cam.RealsenseCam()
                self.thermal = thermal_cam.ThermalCam()
            # Prepare variables for camera matrix and distortion coefficients
            self.cmtx_rs = np.zeros(3)
            self.dist_rs = 0
            self.cmtx_thermal = np.zeros(3)
            self.dist_thermal = 0
            self.rmtx = np.zeros(3)
            self.tmtx = np.zeros((3, 1))
            self.essential_mtx = np.zeros(3)
            self.fundamental_mtx = np.zeros(3)
            self.per_view_errors = np.zeros((self.num, 1))
            # Variable for the path to store the calibration config in
            self.path = kwargs.get('path')
            self.shape_x = 0
            self.shape_y = 0
        else:  # mode = 1 -> calibration found, just loading in the important bits
            # Variables for camera matrix and distortion coefficients given via kwargs
            self.cmtx_rs = kwargs.get('cmtx_rs')
            self.dist_rs = kwargs.get('dist_rs')
            self.cmtx_thermal = kwargs.get('cmtx_thermal')
            self.dist_thermal = kwargs.get('dist_thermal')
            self.rmtx = kwargs.get('rmtx')
            self.tmtx = kwargs.get('tmtx')
            self.essential_mtx = kwargs.get('essential_mtx')
            self.fundamental_mtx = kwargs.get('fundamental_mtx')
            self.per_view_errors = kwargs.get('per_view_errors')
            if 'cam' in kwargs and kwargs.get('cam') == 1:
                # Create camera object
                self.rs = realsense_cam.RealsenseCam()
                self.thermal = thermal_cam.ThermalCam()
            self.shape_x = 0
            self.shape_y = 0

    def take_pictures(self, scale_fac):
        while self.found < self.num:
            # Try to get the data from both cameras before doing any calculations with them
            _thermal_img = self.thermal.get_frame()
            # _rs_image, _ = self.rs.get_frame()
            _rs_image, _ = self.rs.get_frame()
            self.shape_x, self.shape_y, _ = _rs_image.shape
            # Turn the rs_image into grayscale and the thermal_img into an 8-bit image for further processing
            _rs_image_gray = cv2.cvtColor(_rs_image, cv2.COLOR_BGR2GRAY)
            _rs_image_gray_resized = _rs_image_gray[:, 80:560]
            _thermal_img_gray = thermal_cam.raw_to_8bit(_thermal_img)
            _thermal_img_gray_resized = cv2.resize(_thermal_img_gray, (0, 0), fx=scale_fac, fy=scale_fac,
                                                   interpolation=cv2.INTER_CUBIC)

            # Find the circle grids for both the rs and thermal image
            _ret_rs, _corners_rs = cv2.findCirclesGrid(_rs_image_gray_resized,
                                                       (self.pattern_rows, self.pattern_columns),
                                                       flags=cv2.CALIB_CB_ASYMMETRIC_GRID, )
            _ret_thermal, _corners_thermal = cv2.findCirclesGrid(_thermal_img_gray_resized,
                                                                 (self.pattern_rows, self.pattern_columns),
                                                                 flags=cv2.CALIB_CB_ASYMMETRIC_GRID)

            if _ret_rs & _ret_thermal:
                self.found += 1

                # Save calibration images prior to drawing the found circles
                _filename_rs = self.path + "rs_" + str(self.found) + ".jpg"
                cv2.imwrite(_filename_rs, _rs_image_gray_resized)
                _filename_thermal = self.path + "thermal_" + str(self.found) + ".jpg"
                cv2.imwrite(_filename_thermal, _thermal_img_gray_resized)

                # Draw and display the _corners.
                _img_with_keypoints_rs = cv2.drawChessboardCorners(_rs_image_gray_resized,
                                                                   (self.pattern_rows, self.pattern_columns),
                                                                   _corners_rs,
                                                                   _ret_rs)
                _img_with_keypoints_thermal = cv2.drawChessboardCorners(_thermal_img_gray_resized,
                                                                        (self.pattern_rows, self.pattern_columns),
                                                                        _corners_thermal,
                                                                        _ret_thermal)

                # Save the calibration images - the grayscale image and the equivalent image with the circle grid marked
                _filename_rs = self.path + "rs_grid_" + str(self.found) + ".jpg"
                cv2.imwrite(_filename_rs, _img_with_keypoints_rs)
                _filename_thermal = self.path + "thermal_grid_" + str(self.found) + ".jpg"
                cv2.imwrite(_filename_thermal, _img_with_keypoints_thermal)

                # Increment found
                print("Status stereo: %d out of %d pictures taken." % (self.found, self.num))

            try:
                # display both images with marked circles
                # img_thermal = cv2.resize(_img_with_keypoints_thermal, _img_with_keypoints_rs.shape)
                # img = cv2.hconcat(_img_with_keypoints_rs, img_thermal)
                cv2.imshow("img_rs", _img_with_keypoints_rs)
                cv2.imshow("img_thermal", _img_with_keypoints_thermal)
                # cv2.imshow('img_keypoints', img)
                _img_with_keypoints_rs = None
                _img_with_keypoints_thermal = None
                cv2.waitKey(1000)
            except:
                # Else, just display the grayscale images
                # img_thermal = cv2.resize(_thermal_img_gray, _rs_image_gray.shape)
                # img = cv2.hconcat(_rs_image_gray, img_thermal)
                cv2.imshow("img_rs", _rs_image_gray_resized)
                cv2.imshow("img_thermal", _thermal_img_gray_resized)
                # cv2.imshow('img_gray',img)
                cv2.waitKey(100)
        self.thermal.stop_stream
        self.rs.stop_pipeline

    def calibrate(self, mtx_rs, mtx_thermal, dist_coeff_rs, dist_coeff_thermal):
        # Take a random selection of images that have been taken before
        _list_file_names = [f for f in os.listdir(self.path) if f.startswith("thermal_grid_")]
        index = []
        for file in _list_file_names:
            index.append(int(file[13:-4]))
        random.sample(index, self.num)
        index.sort()
        print(index)
        for i in range(self.num):
            # Load previously taken pictures
            _filename_rs = self.path + "rs_" + str(index[i]) + ".jpg"
            _rs_image_gray = cv2.imread(_filename_rs, cv2.IMREAD_GRAYSCALE)
            _filename_thermal = self.path + "thermal_" + str(index[i]) + ".jpg"
            _thermal_img_gray_resized = cv2.imread(_filename_thermal, cv2.IMREAD_GRAYSCALE)

            _ret_rs, _corners_rs = cv2.findCirclesGrid(_rs_image_gray,
                                                       (self.pattern_rows, self.pattern_columns),
                                                       flags=cv2.CALIB_CB_ASYMMETRIC_GRID, )
            _ret_thermal, _corners_thermal = cv2.findCirclesGrid(_thermal_img_gray_resized,
                                                                 (self.pattern_rows, self.pattern_columns),
                                                                 flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
            print(_ret_rs & _ret_thermal)
            if _ret_rs & _ret_thermal:
                # Save imagepoints and objectpoints
                self.imgpoints_rs.append(_corners_rs)
                self.imgpoints_thermal.append(_corners_thermal)
                self.objpoints.append(self.objp)  # Certainly, every loop objp is the same, in 3D.

        try:
            # All pictures analyzed, time for stereo calibration
            retval, self.cmtx_rs, self.dist_rs, self.cmtx_thermal, self.dist_thermal, self.rmtx, self.tmtx, self.essential_mtx, _, _, self.fundamental_mtx, self.per_view_errors = cv2.stereoCalibrateExtended(
                    objectPoints=self.objpoints,
                    imagePoints1=self.imgpoints_rs,
                    imagePoints2=self.imgpoints_thermal,
                    cameraMatrix1=np.float32(mtx_rs),
                    distCoeffs1=np.float32(dist_coeff_rs),
                    cameraMatrix2=np.float32(mtx_thermal),
                    distCoeffs2=np.float32(dist_coeff_thermal),
                    imageSize=(self.shape_x, self.shape_y),
                    criteria=self.termin_crit,
                    flags=cv2.CALIB_FIX_INTRINSIC,
                    R=np.float32(self.rmtx),
                    T=np.float32(self.tmtx)
                )
            #print(retval, self.cmtx_rs, self.dist_rs, self.cmtx_thermal, self.dist_thermal, self.rmtx, self.tmtx, self.essential_mtx, self.fundamental_mtx, self.per_view_errors)
        except Exception as e:
            print(e)
            traceback.print_exc
            print("Calibration failed")
            exit(-1)

        if retval:
            # Calculate reprojection error:
            _y_rs = self.per_view_errors[::2]
            _y_thermal = self.per_view_errors[1::2]

            reprojection_error_av_rs = np.array(np.nanmean(_y_rs))
            reprojection_error_sd_rs = np.array(np.nanstd(_y_rs))

            reprojection_error_av_thermal = np.array(np.nanmean(_y_thermal))
            reprojection_error_sd_thermal = np.array(np.nanstd(_y_thermal))

            print(f"Stereo reprojection error RealSense: {reprojection_error_av_rs} +/- {reprojection_error_sd_rs}\n")
            print(f"Stereo reprojection error Thermal: {reprojection_error_av_thermal} +/- {reprojection_error_sd_thermal}\n")
            print('Stereo calibration done.')

            # Display reprojection error as pyplot
            _x = np.linspace(1, len(self.per_view_errors), len(self.per_view_errors))
            fig, ax = plt.subplots(nrows=1, ncols=1)
            self.per_view_errors = [item for sub_list in self.per_view_errors for item in sub_list]

            #print(_x)
            #print(self.per_view_errors)
            ax.set_title('Mean Reprojection Error per Image %d Samples' % self.num)
            ax.set_xlabel('Images')
            ax.set_ylabel('Mean Error in Pixels')
            ax.set_xticks(np.arange(1, 21, step=1))
            _y_rs = self.per_view_errors[0::2]
            _y_thermal = self.per_view_errors[1::2]
            ax.bar(_x-0.2, _y_rs, width=0.4, align='center', label='Intel® RealSense™ Depth Camera D455')
            ax.bar(_x+0.2, _y_thermal, width=0.4, align='center', label='PureThermal Mini Pro JST-SR')
            ax.axhline(reprojection_error_av_rs, alpha=0.5, dashes=(5, 5),
                       label='Overall Mean Error RealSense: %.2f' % reprojection_error_av_rs, color='blue')
            ax.axhline(reprojection_error_av_thermal, alpha=0.5, dashes=(5, 5),
                       label='Overall Mean Error Thermal: %.2f' % reprojection_error_av_thermal, color='orange')
            ax.legend()
            fig.savefig(self.path + 'stereo_reprojection_error_' + str(self.num) + '.png')

            # Save the calibration values
            data = {'camera_matrix_rs': np.asarray(self.cmtx_rs).tolist(),  # np.asarray(cmtx_rs).tolist(),
                    'dist_coeff_rs': np.asarray(self.dist_rs).tolist(),  # np.asarray(dist_rs).tolist(),
                    'camera_matrix_thermal': np.asarray(self.cmtx_thermal).tolist(),  # np.asarray(cmtx_thermal).tolist(),
                    'dist_coeff_thermal': np.asarray(self.dist_thermal).tolist(),  # np.asarray(dist_thermal).tolist(),
                    'R_matrix': np.asarray(self.rmtx).tolist(),  # np.asarray(rmtx).tolist(),
                    't_matrix': np.asarray(self.tmtx).tolist(),  # np.asarray(tmtx).tolist(),
                    'essential_matrix': np.asarray(self.essential_mtx).tolist(),  # np.asarray(essential_mtx).toList(),
                    'fundamental_matrix': np.asarray(self.fundamental_mtx).tolist(),  # np.asarray(fundamental_mtx).tolist()
                    'reprojection_error': np.asarray(self.per_view_errors).tolist()
                    }

            with open(self.path + "calibration_stereo_" + str(self.num) +".yaml", "w") as file:
                yaml.dump(data, file)

            # Read YAML file
            # with open(CalibrationFileName, 'r') as stream:
            #    dictionary = yaml.safe_load(stream)
            # camera_matrix = dictionary.get("camera_matrix")
            # dist_coeffs = dictionary.get("dist_coeff")
            # rvecs = dictionary.get("rvecs")
            # tvecs = dictionary.get("tvecs")

