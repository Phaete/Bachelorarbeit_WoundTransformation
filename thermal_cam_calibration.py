# This script is part of the bachelor thesis
# "Localization of medical measurement data from injured persons for use in rescue robotics"
# from Konstantin Wenig written for the university to luebeck's bachelor degree in robotics and autonomous systems
# Redistribution and change without consent it not allowed

import cv2
import numpy as np
import yaml
import thermal_cam
import traceback
import random
import os
from matplotlib import pyplot as plt

try:
    from queue import Queue
except ImportError:
    from queue import Queue


# Convert 16-bit to 8-bit
def raw_to_8bit(data):
    _data_norm = cv2.normalize(data, None, 0, 256, cv2.NORM_MINMAX)
    return cv2.cvtColor(np.uint8(_data_norm), cv2.COLOR_GRAY2RGB)


class ThermalCalibration:
    def __init__(self, **kwargs):
        if kwargs.get('mode') == 0:  # args = 0 -> no calibration found or being repeated
            # Variables for the camera calibration
            # Including variables for the calibration object
            if 'pattern' in kwargs and kwargs.get('pattern') == 'chessboard':
                self.pattern = 'chessboard'
                self.pattern_rows = 4
                self.pattern_columns = 11
                self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
            else:
                self.pattern = 'circle'
                self.pattern_rows = 4
                self.pattern_columns = 11
            #self.distance_world_unit = 1
            # The termination criteria for the calibration
            self.termin_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
            # And arrays to store object points and image points from all the images
            self.objpoints = []  # 3d point in real world space
            self.imgpoints = []  # 2d points in image plane.
            # Preparation for the object points
            self.objp = np.array(
                [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [0.5, 0.5, 0], [1.5, 0.5, 0], [2.5, 0.5, 0], [3.5, 0.5, 0]], np.float32)
            for y in range(2, 11):
                for x in range(4):
                    self.objp = np.append(
                        self.objp, 
                        [np.array([self.objp[4 * (y - 2) + x][0], self.objp[4 * (y - 2) + x][1] + 1, 0], np.float32)], axis=0)
            # Variables for the amount of pictures that should be taken and processed before starting calibration
            self.num = kwargs.get('num')
            self.found = kwargs.get('found')
            if kwargs.get('cam'):
                # Create camera object
                self.thermal = thermal_cam.ThermalCam()
            # Prepare variables for camera matrix and distortion coefficients
            self.mtx = np.zeros((3, 3))
            self.dist = np.zeros(5)
            self.rvecs = np.zeros((3, 3))
            self.tvecs = np.zeros(3)
            self.per_view_errors = np.zeros(self.num)
            # Prepare path variable to store calibration file in
            self.path = kwargs.get('path')
        elif kwargs.get('mode') == 1:  # args = 1 -> calibration found, just loading in the important bits
            self.mtx = kwargs.get('mtx')
            self.dist = kwargs.get('dist')
            self.rvecs = kwargs.get('rvecs')
            self.tvecs = kwargs.get('tvecs')
        else:
            # Create camera object
            self.thermal = thermal_cam.ThermalCam()
            self.mtx = kwargs.get('mtx')
            self.dist = kwargs.get('dist')
            self.rvecs = kwargs.get('rvecs')
            self.tvecs = kwargs.get('tvecs')

    def take_pictures(self, **kwargs):
        # Extract scale factor from kwargs:
        if 'scale_fac' in kwargs:
            scale_fac = kwargs.get('scale_fac')
        else:
            scale_fac = 8
        # Loop to take pictures
        if self.pattern == 'circle':
            while self.found < self.num:  # Here, self.num is the amount of pictures to be taken
                _img_data = self.thermal.get_frame()
                #_img_gray = cv2.cvtColor(_img_data, cv2.COLOR_BGR2GRAY)
                _img_gray = raw_to_8bit(_img_data)
                _img_gray_resized = cv2.resize(_img_gray, (0, 0), fx=scale_fac, fy=scale_fac, interpolation=cv2.INTER_CUBIC)

                ret, _corners = cv2.findCirclesGrid(_img_gray_resized, (self.pattern_rows, self.pattern_columns),
                                                    flags=cv2.CALIB_CB_ASYMMETRIC_GRID  # ,
                                                    # blobDetector=self.blobDetector
                                                    )  # Find the circle grid

                if ret:
                    self.found += 1

                    # Save calibration images prior to drawing the found circles
                    _filename = self.path + "thermal_" + str(self.found) + ".jpg"
                    cv2.imwrite(_filename, _img_gray_resized)

                    # Draw and display the _corners.
                    _img_with_keypoints = cv2.drawChessboardCorners(_img_gray_resized,
                                                                    (self.pattern_rows, self.pattern_columns), _corners,
                                                                    ret)

                    # Save the calibration images - the grayscale image and the equivalent image with the circle grid marked
                    _filename = self.path + "thermal_grid_" + str(self.found) + ".jpg"
                    cv2.imwrite(_filename, _img_with_keypoints)


                    # Increment found
                    print("Status thermal: %d out of %d pictures taken." % (self.found, self.num))

                try:
                    # display img_with_keypoints if the circlegrid could be detected
                    cv2.imshow("img", _img_with_keypoints)
                    _img_with_keypoints = None
                    cv2.waitKey(1000)
                except:
                    # Else, just display _img_gray
                    cv2.imshow("img", _img_gray_resized)  # display
                    cv2.waitKey(100)

        else:
            while self.found < self.num:  # Here, self.num is the amount of pictures to be taken
                _img_data = self.thermal.get_frame()
                #_img_gray = cv2.cvtColor(_img_data, cv2.COLOR_BGR2GRAY)
                _img_gray = raw_to_8bit(_img_data)
                _img_gray_resized = cv2.resize(_img_gray, (0, 0), fx=scale_fac, fy=scale_fac, interpolation=cv2.INTER_CUBIC)

                ret, _corners = cv2.findChessboardCorners(_img_gray_resized,
                                                          (self.pattern_rows, self.pattern_columns),
                                                          None)
                if ret:
                    self.found += 1

                    # Save calibration images prior to drawing the found circles
                    _filename = self.path + "thermal_cb_" + str(self.found) + ".jpg"
                    cv2.imwrite(_filename, _img_gray_resized)

                    # Refine corners
                    _corners_refined = cv2.cornerSubPix(_img_gray_resized, _corners, (3, 3), (-1, -1), self.criteria)

                    # Draw and display the _corners.
                    _img_with_keypoints = cv2.drawChessboardCorners(_img_gray_resized,
                                                                    (self.pattern_rows, self.pattern_columns),
                                                                    _corners_refined, ret)

                    # Save the calibration images - the grayscale image and the equivalent image with the circle grid
                    # marked
                    _filename = self.path + "thermal_cb_grid_" + str(self.found) + ".jpg"
                    cv2.imwrite(_filename, _img_with_keypoints)

                    # Increment found
                    print("Status thermal: %d out of %d pictures taken." % (self.found, self.num))

                try:
                    # display img_with_keypoints if the circlegrid could be detected
                    cv2.imshow("img", _img_with_keypoints)
                    _img_with_keypoints = None
                    cv2.waitKey(1000)
                except:
                    # Else, just display _img_gray
                    cv2.imshow("img", _img_gray_resized)  # display
                    cv2.waitKey(100)


        # When everything done, release the thermal camera and close all windows from opencv that are still open
        print("All pictures taken.")
        self.thermal.stop_stream()
        cv2.destroyAllWindows()

    def calibrate(self):
        if self.pattern == 'circle':
            # Take a random selection of images that have been taken before
            _list_file_names = [f for f in os.listdir(self.path) if f.startswith("thermal_grid_")]
            index = []
            print(_list_file_names)
            # Extract their indices
            for file in _list_file_names:
                index.append(int(file[13:-4]))
        else:
            # Take a random selection of images that have been taken before
            _list_file_names = [f for f in os.listdir(self.path) if f.startswith("thermal_cb_grid_")]
            index = []
            print(_list_file_names)
            # Extract their indices
            for file in _list_file_names:
                index.append(int(file[16:-4]))
        # From that list, extract a random sample based on the previously configured amount of images to be used
        # for the calibration
        index = random.sample(index, self.num)
        index.sort()
        _list_good_images = []
        if self.pattern == 'circle':
            for i in range(self.num):
                # Load previously taken pictures
                _filename = self.path + "thermal_" + str(index[i]) + ".jpg"
                _img_gray_resized = cv2.imread(_filename, cv2.IMREAD_GRAYSCALE)

                ret, _corners = cv2.findCirclesGrid(_img_gray_resized, (self.pattern_rows, self.pattern_columns),
                                                    flags=cv2.CALIB_CB_ASYMMETRIC_GRID  # ,
                                                    # blobDetector=self.blobDetector
                                                    )  # Find the circle grid
                print(ret)
                if ret:
                    # Add found imgpoints and the predefined objpoints to the arrays
                    self.imgpoints.append(_corners)
                    self.objpoints.append(self.objp)  # Certainly, every loop objp is the same, in 3D.
                    _list_good_images.append(index[i])
        else:
            for i in range(self.num):
                # Load previously taken pictures
                _filename = self.path + "thermal_cb_" + str(index[i]) + ".jpg"
                _img_gray_resized = cv2.imread(_filename, cv2.IMREAD_GRAYSCALE)

                ret, _corners = cv2.findChessboardCorners(_img_gray_resized,
                                                          (self.pattern_rows, self.pattern_columns),
                                                          None)

                _corners_refined = cv2.cornerSubPix(_img_gray_resized, _corners, (3, 3), (-1, -1), self.criteria)

                print(ret)
                if ret:
                    # Add found imgpoints and the predefined objpoints to the arrays
                    self.imgpoints.append(_corners_refined)
                    self.objpoints.append(self.objp)  # Certainly, every loop objp is the same, in 3D.
                    _list_good_images.append(index[i])
        print(_list_good_images)

        # Calibrate the thermal camera with the previously generated imgpoints and objpoints
        try:
            ret_calib, self.mtx, self.dist, self.rvecs, self.tvecs, stdDevInt, stdDevExt, self.per_view_errors = cv2.calibrateCameraExtended(
                self.objpoints, self.imgpoints, _img_gray_resized.shape[::-1], None, None
            )

        except Exception as e:
            print(e)
            traceback.print_exc
            print("Calibration failed")
            exit(-1)

        if ret_calib:
            # Calculate reprojection error:
            reprojection_error_av = np.array(np.nanmean(self.per_view_errors))
            reprojection_error_sd = np.array(np.nanstd(self.per_view_errors))

            print(f"Thermal reprojection error: {reprojection_error_av} +/- {reprojection_error_sd}\n")
            print('Thermal calibration done.')

            # Display reprojection error as pyplot
            _x = np.linspace(1, len(self.per_view_errors), len(self.per_view_errors))
            fig, ax = plt.subplots(nrows=1, ncols=1)
            self.per_view_errors = [item for sub_list in self.per_view_errors for item in sub_list]
            #print(_x)
            #print(self.per_view_errors)
            ax.set_title('Mean Reprojection Error per Image')
            ax.set_xlabel('Images')
            ax.set_ylabel('Mean Error in Pixels')
            ax.bar(_x, self.per_view_errors)
            ax.axhline(reprojection_error_av, alpha=0.5, dashes=(5, 5), label='Overall Mean Error: %.2f' % reprojection_error_av)
            ax.axhline(reprojection_error_av+reprojection_error_sd, alpha=0.5, dashes=(5, 5),
                       label='Standard Deviation: +/- %.2f' % reprojection_error_sd, color='red')
            ax.axhline(reprojection_error_av - reprojection_error_sd, alpha=0.5, dashes=(5, 5),
                       color='red')
            ax.legend()
            if self.pattern == 'circle':
                fig.savefig(self.path + 'thermal_reprojection_error_' + str(self.num) + '.png')
            else:
                fig.savefig(self.path + 'thermal_cb_reprojection_error_' + str(self.num) + '.png')

        if self.pattern == 'circle':
            for k in range(len(self.objpoints)):
                # Load previously taken pictures
                _filename = self.path + "thermal_" + str(_list_good_images[k]) + ".jpg"
                _img_gray = cv2.imread(_filename, cv2.IMREAD_GRAYSCALE)

                ret, _corners = cv2.findCirclesGrid(_img_gray, (self.pattern_rows, self.pattern_columns),
                                                    flags=cv2.CALIB_CB_ASYMMETRIC_GRID  # ,
                                                    # blobDetector=self.blobDetector
                                                    )  # Find the circle grid

                if ret:
                    # Reproject 3D points onto an image plane
                    im_points, jac = cv2.projectPoints(self.objpoints[k], self.rvecs[k], self.tvecs[k], self.mtx, self.dist)
                    # print(im_points)

                    # Draw and display the _corners.
                    _img_with_keypoints = cv2.drawChessboardCorners(_img_gray,
                                                                    (self.pattern_rows, self.pattern_columns), _corners,
                                                                    ret)

                    # Convert to BGR for reprojected points showing in a different color
                    _img_with_keypoints = cv2.cvtColor(_img_with_keypoints, cv2.COLOR_GRAY2BGR)

                    # Draw all the reprojected points as circles onto the image with the marked grid
                    for im_point in im_points:
                        _img_with_keypoints = cv2.circle(_img_with_keypoints,
                                                         (int(im_point[0][0]), int(im_point[0][1])), 10, (0, 0, 255), 2)

                    # Add the total reprojection error as text onto the picture
                    _img_with_keypoints = cv2.putText(_img_with_keypoints, 'Mean Error: %.3f Pixel' % self.per_view_errors[k],
                                                      (5, 15), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))

                    # Save calibration image with the circle grid marked and the reprojection
                    _filename = self.path + "thermal_reprojected_" + str(_list_good_images[k]) + ".jpg"
                    cv2.imwrite(_filename, _img_with_keypoints)
        else:
            for k in range(len(self.objpoints)):
                # Load previously taken pictures
                _filename = self.path + "thermal_cb_" + str(_list_good_images[k]) + ".jpg"
                _img_gray = cv2.imread(_filename, cv2.IMREAD_GRAYSCALE)

                ret, _corners = cv2.findChessboardCorners(_img_gray,
                                                          (self.pattern_rows, self.pattern_columns),
                                                          None)

                _corners_refined = cv2.cornerSubPix(_img_gray, _corners, (3, 3), (-1, -1), self.criteria)

                if ret:
                    # Reproject 3D points onto an image plane
                    im_points, jac = cv2.projectPoints(self.objpoints[k], self.rvecs[k], self.tvecs[k], self.mtx,
                                                       self.dist)

                    # Draw and display the _corners.
                    _img_with_keypoints = cv2.drawChessboardCorners(_img_gray,
                                                                    (self.pattern_rows, self.pattern_columns),
                                                                    _corners_refined, ret)

                    # Convert to BGR for reprojected points showing in a different color
                    _img_with_keypoints = cv2.cvtColor(_img_with_keypoints, cv2.COLOR_GRAY2BGR)

                    # Draw all the reprojected points as circles onto the image with the marked grid
                    for im_point in im_points:
                        _img_with_keypoints = cv2.circle(_img_with_keypoints,
                                                         (int(im_point[0][0]), int(im_point[0][1])), 10, (0, 0, 255), 2)

                    # Add the total reprojection error as text onto the picture
                    _img_with_keypoints = cv2.putText(_img_with_keypoints, 'Mean Error: %.3f Pixel' % self.per_view_errors[k],
                                                      (5, 15), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))

                    # Save calibration image with the circle grid marked and the reprojection
                    _filename = self.path + "thermal_cb_reprojected_" + str(_list_good_images[k]) + ".jpg"
                    cv2.imwrite(_filename, _img_with_keypoints)

        # And then save the calibration values
        data = {'camera_matrix': np.asarray(self.mtx).tolist(),
                'dist_coeff': np.asarray(self.dist).tolist(),
                'rvecs': np.asarray(self.rvecs).tolist(),
                'tvecs': np.asarray(self.tvecs).tolist(),
                'reprojection_error': np.asarray(self.per_view_errors).tolist()}

        if self.pattern == 'circle':
            with open(self.path + "calibration_thermal_" + str(self.num) + ".yaml", "w") as file:
                yaml.dump(data, file)
        else:
            with open(self.path + "calibration_thermal_cb.yaml", "w") as file:
                yaml.dump(data, file)

        # Read YAML file
        # with open(CalibrationFileName, 'r') as stream:
        #    dictionary = yaml.safe_load(stream)
        # camera_matrix = dictionary.get("camera_matrix")
        # dist_coeffs = dictionary.get("dist_coeff")
        # rvecs = dictionary.get("rvecs")
        # tvecs = dictionary.get("tvecs")

