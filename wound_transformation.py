# This script is part of the bachelor thesis
# "Localization of medical measurement data from injured persons for use in rescue robotics"
# from Konstantin Wenig written for the university to luebeck's bachelor degree in robotics and autonomous systems
# Redistribution and change without consent it not allowed

import cv2
import numpy as np
import yaml
import os
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt


def visualize(image, wounds):
    # Load image
    _image = image
    for wound in wounds:
        _image = cv2.circle(_image, (np.round(wound[0]).astype('int'), np.round(wound[1]).astype('int')), 5,
                            (0, 0, 255), 2)

    return _image


def project_point_onto_vector(point, end, start):
    # Projection only requires 2D vectors
    _start = start[:2]
    _end = end[:2]
    _point = point[:2]

    # Calculate the vector from start to end
    _vector = _end - _start

    # Calculate the vector from start to point
    _point_vector = _point - _start

    # Calculate the projection
    # The formula for the projection is proj(u) v = (v dot u / abs(u)^2) * u and will result in a scalar projection of
    # the point onto the vector. If we apply that scalar to the direction vector _vector and start from point _start
    # we receive the projected point _projection onto the vector _vector
    _projection_vec = (np.dot(_vector, _point_vector) / (np.linalg.norm(_vector) ** 2)) * _vector
    _projection = _start + _projection_vec
    # If the points given to this function are 3-dimensional, calculate the z coordinate of the projected point based
    # on the given parameters using the parametric equations
    if len(start.shape) == 3:
        # _vector and _projection_vec are direction vectors that can start at (0, 0, 0) to make the calculation easier
        # Because of this and _vector & _projection_vec pointing in the same direction, we can write _vector as
        # _vector = lambda * _projection_vec
        # with lambda = [_a, _b, _c] and _a, _b, _c should all be equal. We consider floating point errors with the last
        # check.
        _a = _projection_vec[0] / _vector[0]
        _b = _projection_vec[1] / _vector[1]
        if _b - _a < np.finfo(np.float32).eps:
            _projection.append(_start[2] + _vector[2] * _a)
    # print('Projection: ', _projection)
    # Check if the projection is within the segment defined by start and end
    if not point_in_vector(_vector, _projection_vec):
        raise ValueError()

    return _projection


def point_in_vector(vector, projection):
    # Check only requires 2D vectors
    _vector = vector[:2]
    _projection = projection[:2]
    dot_p = np.dot(_vector, _projection)
    return 0 <= dot_p <= np.linalg.norm(_vector) ** 2


def point_pos_relative_to_vector(point, end, start):
    # Relative position only requires 2D vectors
    _start = start[:2]
    _end = end[:2]
    _point = point[:2]

    # Calculate the vector from start to end
    _vector = _end - _start

    # Calculate the vector from start to point
    _point_vector = _point - _start

    # Calculate the cross product
    _cross_product = np.cross(_vector, _point_vector)

    # Check the sign of the cross product
    if _cross_product > 0:
        return "left"
    elif _cross_product < 0:
        return "right"
    else:
        return "on"


def create_new_point_on_same_side(ratio, rel_pos, end, start, distance, ratio_triangle):
    # Calculate the vector from start to end
    _vector = end - start

    # Project a point onto a vector based on the ratio "start->wound / start->end"
    _projection = start + _vector * ratio

    # Rotate the vector by 90 degrees in the direction of relative_position

    if rel_pos == "left":  # +90°
        _rotated_vector = np.array([-_vector[1], _vector[0]])
    elif rel_pos == "right":  # -90°
        _rotated_vector = np.array([_vector[1], -_vector[0]])
    else:
        return None

    # Normalize the rotated vector and scale it by distance but keep the ratio
    _rotated_vector = _rotated_vector / np.linalg.norm(_rotated_vector) * distance * ratio_triangle

    # Calculate the new point by adding the rotated vector to the projection and return it
    _new_point = _projection[:2] + _rotated_vector

    return _new_point


def calc_ratio(point, end, start):
    # Calculate the vector from start to end
    _vector = end - start

    # Calculate the vector from start to point
    _point_vector = point - start

    # Calculate the length of the vectors
    _len_vector = np.linalg.norm(_vector)
    _len_point_vector = np.linalg.norm(_point_vector)

    # Return ratio between those two
    return _len_point_vector / _len_vector


def calc_ratio_triangle(end, start, vec_standard):
    # Calculate the length of the vectors
    _len_vector_before = np.linalg.norm(end - start)
    _len_vector_after = np.linalg.norm(vec_standard)

    # Return ratio between those two
    return _len_vector_after / _len_vector_before


def calculate_reprojection_error(wounds, reprojected_wounds):
    error_per_wound = []
    # print(reprojected_wounds)
    for i in range(len(wounds)):
        try:
            if not (reprojected_wounds[i][0] < 0 and reprojected_wounds[i][1] < 0):
                error_per_wound.append(np.linalg.norm(wounds[i] - reprojected_wounds[i]))
            else:
                error_per_wound.append(-1)
        except:
            error_per_wound.append((-1))
    return error_per_wound


class WoundTransformation:
    def __init__(self, **kwargs):
        self.base_path = os.path.dirname(
            os.path.abspath("preOpenPose.py")
        )
        # Load skeleton data
        if 'skeleton' in kwargs and kwargs.get('skeleton') == 'nuitrack':
            print('NuiTrack not yet implemented.')
            exit(1)
        else:
            # Load openpose data
            # OpenPose Joint Pairs
            self.pose_pairs = [
                [0, 1], [0, 15], [0, 16], [15, 17], [16, 18],  # Head
                [1, 2], [2, 3], [3, 4],  # Right Arm
                [1, 5], [5, 6], [6, 7],  # Left Arm
                [1, 8],  # Pelvis
                [8, 9], [9, 10], [10, 11], [11, 22], [22, 23], [11, 24],  # Right Leg
                [8, 12], [12, 13], [13, 14], [14, 19], [19, 20], [14, 21],  # Left Leg
            ]
            # Experimental: Calculated ratios between the length of a limb and its width
            # The length of a limb is np.linalg.norm(_end-_start)
            # The width of a limb is the maximum distance from the vector to a projection so the wound is inside of a
            # limb's area
            # ratio = w/l -> l = w / ratio
            self._pose_pair_ratio = {
                '[0, 1]': 2,
                '[1, 2]': 1,
                '[2, 3]': 2.5,
                '[3, 4]': 2.5,
                '[1, 5]': 1,
                '[5, 6]': 2.5,
                '[6, 7]': 2.5,
                '[1, 8]': 1.25,
                '[8, 9]': 0.4,
                '[9, 10]': 2.5,
                '[10, 11]': 4,
                '[11, 22]': 1.5,
                '[8, 12]': 0.4,
                '[12, 13]': 2.5,
                '[13, 14]': 4,
                '[14, 19]': 1.5
            }

            # OpenPose Joint Descriptions
            pose_descriptions = [
                "Nose", "Neck", "Right Shoulder", "Right Elbow", "Right Wrist",
                "Left Shoulder", "Left Elbow", "Left Wrist", "MidHip",
                "Right Hip", "Right Knee", "Right Ankle", "Left Hip",
                "Left Knee", "Left Ankle", "Right Eye", "Left Eye",
                "Right Ear", "Left Ear", "Left Big Toe", "Left Small Toe",
                "Left Heel", "Right Big Toe", "Right Small Toe", "Right Heel",
                "Background"
            ]
            # OpenPose keypoints for standard view
            with open(self.base_path + '/assets/pictures/standard/male_100_keypoints.json', 'r') as stream:
                _data = yaml.safe_load(stream)
                _keypoints_standard_2d = _data['people'][0]['pose_keypoints_2d']
                # Convert the list to a dictionary that consists of entries following (x,y,confidence)
                self.keypoints_standard = {}
                j = 0
                for i in range(len(_keypoints_standard_2d)):
                    if i % 3 == 0:
                        self.keypoints_standard[j] = {"x": _keypoints_standard_2d[i],
                                                      "y": _keypoints_standard_2d[i + 1],
                                                      "c": _keypoints_standard_2d[i + 2]}
                        j = j + 1

        # Load the standard view of the body
        self.standard_view_img = cv2.imread(self.base_path + '/assets/pictures/standard/male_100_rendered.png')

    # Check if the converted wound is inside the approximated area of the limb.
    # This approximation is based on rectangles that have the limb-vectors as center line and their width is based on
    # experiments. The width of the limb is ONLY VALID FOR THE STANDARD male_100_renderer.png and has to be
    # re-calculated for any other standard!
    def inside_limb_area(self, closest_pair):
        pair = list(closest_pair[0])
        distance = closest_pair[1]
        _limb_length = np.linalg.norm(
            np.array([
                self.keypoints_standard[pair[1]]['x'],
                self.keypoints_standard[pair[1]]['y']
            ]) - np.array([
                self.keypoints_standard[pair[0]]['x'],
                self.keypoints_standard[pair[0]]['y']
            ])
        )
        if pair in [[1, 0], [0, 15], [0, 16], [15, 17], [16, 18]]:   # closest pair is in the head region
            _head_width = np.linalg.norm(
                np.array([
                    np.around(self.keypoints_standard[0]['x']).astype('int'),
                    np.around(self.keypoints_standard[16]['y']).astype('int')
                ]) - np.array([
                    np.around(self.keypoints_standard[17]['x']).astype('int'),
                    np.around(self.keypoints_standard[15]['y']).astype('int')
                ])
            ) * 1.3
            print('Distance: ', distance)
            print('Head Width: ', _head_width)
            return distance < _head_width
        elif pair in [[11, 22], [22, 23], [11, 24]]:
            print('Distance: ', distance)
            print('Limb Width: ', _limb_length / self._pose_pair_ratio[str([11, 22])])
            return distance < _limb_length / self._pose_pair_ratio[str([11, 22])]/1.5
        elif pair in [[14, 19], [19, 20], [14, 21]]:
            print('Distance: ', distance)
            print('Limb Width: ', _limb_length / self._pose_pair_ratio[str([14, 19])])
            return distance < _limb_length / self._pose_pair_ratio[str([14, 19])]/1.5
        else:
            print('Distance: ', distance)
            print('Limb Width: ', _limb_length / self._pose_pair_ratio[str(pair)])
            return distance < _limb_length / self._pose_pair_ratio[str(pair)]/1.5

    def locate_wounds(self, image, keypoints, wounds, depth_image, num):
        # Load image
        _image = np.array(image)
        _img_copy = np.array(_image)
        # Load keypoints
        _keypoints = keypoints
        # Load wounds
        _wounds = wounds
        # print('Wounds: ', _wounds)
        _converted_wound_pos = []
        if not depth_image is None:
            _depth = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
        # For each wound, check the closest keypoint
        for wound in _wounds:
            #_image = cv2.circle(_image, (np.round(wound[0]).astype('int'), np.round(wound[1]).astype('int')), 5,
            #                    (0, 0, 255), 2)

            if not depth_image is None:
                wound = np.append(wound, _depth[wound[0]][wound[1]])
            # Variable that saves the distance from the wound to each keypoint pair (pair: distance)
            dist_wound_key_pair = {}
            _projected_wound = [0, 0]
            # Check and log distance to each keypoint
            for pair in self.pose_pairs:
                # If a joint can not be identified with openpose, the (x, y, c) vector of said joint will be (0, 0, 0)
                # Those joints are invalid and will thus be given a "invalid" distance, meaning to high to be considered
                if _keypoints[pair[0]]['c'] == 0 or _keypoints[pair[1]]['c'] == 0:
                    dist_wound_key_pair[(pair[0], pair[1])] = 999999999
                    # print('One or both joints not visible.')
                    continue
                # For the rest of the joints, calculate the projected point
                # This method will throw an exception if the projection is not between the two joint-keypoints, meaning
                # the shortest distance between the projected wound and this keypair is invalid too

                # Try to make code viable with and without depth data
                if not depth_image is None:
                    _end = np.array([
                        _keypoints[pair[1]]['x'],
                        _keypoints[pair[1]]['y'],
                        _depth[int(_keypoints[pair[1]]['x'])][int(_keypoints[pair[1]]['y'])]
                    ])
                    _start = np.array([
                        _keypoints[pair[0]]['x'],
                        _keypoints[pair[0]]['y'],
                        _depth[int(_keypoints[pair[0]]['x'])][int(_keypoints[pair[0]]['y'])]
                    ])
                else:
                    _end = np.array([_keypoints[pair[1]]['x'],
                                     _keypoints[pair[1]]['y']])
                    _start = np.array([_keypoints[pair[0]]['x'],
                                       _keypoints[pair[0]]['y']])
                try:
                    _projected_point = project_point_onto_vector(wound, _end, _start)
                    if not depth_image is None:
                        _projected_point = np.append(_projected_point,
                                                     _depth[int(_projected_point[0])][int(_projected_point[1])])
                    dist_wound_key_pair[(pair[0], pair[1])] = np.linalg.norm(wound - _projected_point)
                # The projected wound is not between the joint-keypoints
                except ValueError:
                    dist_wound_key_pair[(pair[0], pair[1])] = 1000000
                    continue
            # DEBUG
            #self.save_images(before_rgb=_image, num=num)
            # Find the closest pair, meaning the minimum value in the dist_wound_to_key_pair dictionary
            min_dist_wound_key_pair = min(dist_wound_key_pair.values())
            # Check for error: Projected wound is not on any vector
            if min_dist_wound_key_pair == 1000000:
                _converted_wound_pos.append([-1, -1])  # holes in coverage ?
                continue
            # Check for error: Openpose missing limbs
            if min_dist_wound_key_pair == 999999999:
                _converted_wound_pos.append([-2, -2])  # openpose - missing limb
                continue
            # Extract the keypair with the minimum distance
            closest_pair = {key: val for key, val in dist_wound_key_pair.items()
                            if val == min_dist_wound_key_pair}
            # Check for error: wound got filtered from standard view
            if not self.inside_limb_area(closest_pair.copy().popitem()):
                _converted_wound_pos.append([-3, -3])
                continue
            key = next(iter(closest_pair))
            # Now that the closest pair is known, it is possible to find out where exactly the wound is and then
            # transfer it to the standard view
            # Project the wound onto the vector represented by the two keypoints of the closest pair
            if not depth_image is None:
                _end = np.array([_keypoints[key[1]]['x'],
                                 _keypoints[key[1]]['y'],
                                 _depth[int(_keypoints[key[1]]['x'])][
                                     int(_keypoints[key[1]]['y'])]])
                _start = np.array([int(_keypoints[key[0]]['x']),
                                   int(_keypoints[key[0]]['y']),
                                   _depth[int(_keypoints[key[0]]['x'])][
                                       int(_keypoints[key[0]]['y'])]])
            else:
                _end = np.array([int(_keypoints[key[1]]['x']),
                                 int(_keypoints[key[1]]['y'])])
                _start = np.array([int(_keypoints[key[0]]['x']),
                                   int(_keypoints[key[0]]['y'])])
            try:
                _projected_wound = project_point_onto_vector(wound, _end, _start)
            except ValueError:
                _converted_wound_pos.append([-4, -4])  # reprojection not valid
                continue
            # Calculate the distance ratio from start to projected wound in comparison to start to end to be able to
            # accurately transfer the position of the projected wound to the standard view
            _dist_ratio = calc_ratio(
                _projected_wound,
                np.array([_keypoints[key[1]]['x'], _keypoints[key[1]]['y']]),
                np.array([_keypoints[key[0]]['x'], _keypoints[key[0]]['y']])
            )
            # print('Distance Ratio: ', _dist_ratio)
            # Calculate on what side the wound is in comparison to the vector
            _rel_pos_wound = point_pos_relative_to_vector(
                wound,
                np.array([_keypoints[key[1]]['x'], _keypoints[key[1]]['y']]),
                _projected_wound
            )
            # The information gathered so far is the distance between start and projected wound, the ratio from this
            # distance to the distance from keypoint[0] to keypoint[1], the distance from the wound to the projected
            # wound and the relative position the wound has in regard to the vector from keypoint[0] to keypoint[1]
            # Now convert this wound to the standard view
            # Vector from keypoint to keypoint in standard view
            _point_key_s = np.array([self.keypoints_standard[key[0]]['x'], self.keypoints_standard[key[0]]['y']])
            _point_key_e = np.array([self.keypoints_standard[key[1]]['x'], self.keypoints_standard[key[1]]['y']])
            _vec_standard = _point_key_e - _point_key_s
            # Convert wound position from camera view to standard view
            _conv_wound = create_new_point_on_same_side(
                _dist_ratio,
                _rel_pos_wound,
                np.array([_point_key_e[0], _point_key_e[1]]),
                np.array([_point_key_s[0], _point_key_s[1]]),
                closest_pair[key],
                calc_ratio_triangle(
                    np.array([_keypoints[key[1]]['x'], _keypoints[key[1]]['y']]),
                    np.array([_keypoints[key[0]]['x'], _keypoints[key[0]]['y']]),
                    _vec_standard
                )
            )
            if not depth_image is None:
                _conv_wound = np.append(_conv_wound, _depth[int(_conv_wound[0])][int(_conv_wound[1])])
            _converted_wound_pos.append(_conv_wound)
        return np.around(_converted_wound_pos).astype('int')

    def transform_view(self, converted_wounds):
        _overlay = np.array(self.standard_view_img)
        # For each of the converted wounds, get the thermal data and save it in _overlay
        for wound in converted_wounds:
            if not (wound[0] < 0 and wound[1] < 0):
                _overlay = cv2.circle(_overlay, (np.round(wound[0]).astype('int'), np.round(wound[1]).astype('int')), 5,
                                      (0, 0, 255), 2)
        cv2.imshow("standard view", _overlay)
        cv2.waitKey(10)
        return _overlay

    def save_images(self, **kwargs):
        # Save the images before and after transformation to the "images" folder as output
        if 'num' in kwargs:
            if kwargs.get('num') is None:
                return
            _num = kwargs.get('num')
        else:
            _num = 0
        if 'before_rgb' in kwargs:
            _filename = self.base_path + "/assets/images/" + str(_num).zfill(5) + "_before_rgb.jpg"
            cv2.imwrite(_filename, kwargs.get('before_rgb'))
        if 'before_depth' in kwargs:
            _filename = self.base_path + "/assets/images/" + str(_num).zfill(5) + "_before_depth.jpg"
            cv2.imwrite(_filename, kwargs.get('before_depth'))
        if 'before_thermal' in kwargs:
            _filename = self.base_path + "/assets/images/" + str(_num).zfill(5) + "_before_thermal.jpg"
            cv2.imwrite(_filename, kwargs.get('before_thermal'))
        if 'after' in kwargs:
            _filename = self.base_path + "/assets/images/" + str(_num).zfill(5) + "_after.jpg"
            cv2.imwrite(_filename, kwargs.get('after'))

    def reproject_wounds(self, keypoints, wounds, converted_wounds):
        # Load keypoints
        _keypoints = keypoints
        # Load wounds
        _wounds = wounds
        _converted_wounds = converted_wounds
        _reprojected_wounds = []
        # For every wound in _converted_wounds, do the same procedure as before, but in reverse
        for wound in _converted_wounds:
            wound = wound[:2]
            if wound[0] < 0 and wound[1] < 0:
                _reprojected_wounds.append([-1, -1])
                continue
            # Variable that saves the distance from the wound to each keypoint pair (pair: distance)
            dist_wound_key_pair = {}
            _projected_wound = [0, 0]
            # Check and log distance to each keypoint
            for pair in self.pose_pairs:
                if _keypoints[pair[0]]['c'] == 0 or _keypoints[pair[1]]['c'] == 0:
                    dist_wound_key_pair[(pair[0], pair[1])] = 999999999
                    # print('One or both joints not visible.')
                    continue

                _vec_0 = np.array([self.keypoints_standard[pair[0]]['x'],
                                   self.keypoints_standard[pair[0]]['y']])
                _vec_1 = np.array([self.keypoints_standard[pair[1]]['x'],
                                   self.keypoints_standard[pair[1]]['y']])
                # Calculate the projected point
                try:
                    _projected_point = project_point_onto_vector(wound,
                                                                 _vec_1,
                                                                 _vec_0
                                                                 )
                    # print('Projected Point: ', _projected_point)
                    dist_wound_key_pair[(pair[0], pair[1])] = np.linalg.norm(_projected_point - wound)
                # The projected wound is not between the joint-keypoints
                except ValueError as e:
                    print(e)
                    dist_wound_key_pair[(pair[0], pair[1])] = 1000000
                    continue
            # Find the closest pair, meaning the minimum value in the dist_wound_to_key_pair dictionary
            # print(dist_wound_key_pair)
            min_dist_wound_key_pair = min(dist_wound_key_pair.values())
            if min_dist_wound_key_pair >= 1000000:
                #print('Wound ', wound, ' could not be transformed.'
                #                       ' Please check the skeleton- and wound-data for validity')
                _reprojected_wounds.append([-1, -1])
                continue
            # Extract the keypair with the minimum distance
            closest_pair = {key: val for key, val in dist_wound_key_pair.items()
                            if val == min_dist_wound_key_pair}
            # print('Closest Pair: ', closest_pair)
            key = next(iter(closest_pair))
            # print('Wound: ', wound)
            # print('Keypair: ', key)
            # print('Closest Pair: ', closest_pair)

            _vec_0 = np.array([self.keypoints_standard[key[0]]['x'], self.keypoints_standard[key[0]]['y']])
            _vec_1 = np.array([self.keypoints_standard[key[1]]['x'], self.keypoints_standard[key[1]]['y']])
            # Now that the closest pair is known, it is possible to find out where exactly the wound is and then
            # transfer it to the standard view
            # Project the wound onto the vector represented by the two keypoints of the closest pair
            _projected_wound = project_point_onto_vector(wound, _vec_1, _vec_0)
            # print('Projected Point: ', _projected_wound)
            # Calculate the distance ratio from start to projected wound in comparison to start to end to be able to
            # accurately transfer the position of the projected wound to the standard view
            _dist_ratio = calc_ratio(_projected_wound, _vec_1, _vec_0)
            # print('Distance Ratio Reprojektion', _dist_ratio)
            # Calculate on what side the wound is in comparison to the vector
            _rel_pos_wound = point_pos_relative_to_vector(
                wound,
                _vec_1,
                _projected_wound
            )
            # print('Relative Wound Position Reprojektion: ', _rel_pos_wound)
            # The information gathered so far is the distance between start and projected wound, the ratio from this
            # distance to the distance from keypoint[0] to keypoint[1], the distance from the wound to the projected
            # wound and the relative position the wound has in regard to the vector from keypoint[0] to keypoint[1]
            # Now convert this wound to the standard view

            # Vector from keypoint to keypoint in standard view
            _point_key_s = np.array([_keypoints[key[0]]['x'], _keypoints[key[0]]['y']])
            _point_key_e = np.array([_keypoints[key[1]]['x'], _keypoints[key[1]]['y']])
            _vec_original = _point_key_e - _point_key_s
            # Convert wound position from camera view to standard view
            _reprojected_wounds.append(
                create_new_point_on_same_side(
                    _dist_ratio,
                    _rel_pos_wound,
                    _point_key_e,
                    _point_key_s,
                    closest_pair[key],
                    calc_ratio_triangle(
                        _vec_1,
                        _vec_0,
                        _vec_original
                    )
                )
            )
        # print('Re-projected Wound Position:', _reprojected_wounds)
        return _reprojected_wounds
        # return np.around(_reprojected_wounds).astype('int')

    def plot_error(self, error_per_wound, **kwargs):
        error_total = np.sum(np.clip(error_per_wound, a_min=0, a_max=None))
        # Display reprojection error as pyplot
        _x = np.linspace(1, len(error_per_wound), len(error_per_wound))
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.suptitle('Wound Transformation - Reprojection Error per Wound')
        ax.set_title(
            'Total Error: %.6f, Mean Error per Frame: %.6f' % (error_total, error_total / len(error_per_wound)))
        ax.set_xlabel('Wound')
        ax.set_ylabel('Error in Pixels')
        ax.set_ylim(top=np.max(error_per_wound) * 1.2)
        bar = ax.bar(_x, np.clip(error_per_wound, a_min=0, a_max=None))
        for i in bar:
            h = i.get_height()
            if h < 0:
                h = 0.2
                # text = 'Could not localize.'
                text = ''
            else:
                text = f'{h:.6f}'
            # ax.text(i.get_x() + i.get_width() / 2, h, text, ha='center', va='bottom')

        # Save the plots
        if 'image' in kwargs:  # kwargs.get('image') contains the name of the image
            fig.savefig(self.base_path + '/assets/images/wound_reprojection_error_' + str(kwargs.get('image')) + '.png')
        if 'video' in kwargs:  # kwargs.get('video') contains the name of the video frame
            fig.savefig(self.base_path + '/assets/videos/wound_reprojection_error_' + str(kwargs.get('video')) + '.png')

        # Display the plot
        # fig.show()  # show not available in agg backend
        plt.close(fig)

    def display_error(self, image, wounds, reprojected_wounds, **kwargs):
        _error_per_wound = calculate_reprojection_error(wounds, reprojected_wounds)
        _error_total = np.sum(np.clip(_error_per_wound, a_min=0, a_max=None))
        # print('Total re-projection error: ', _error_total)
        # print('Error per wounds: ', _error_per_wound)
        # Load image
        _image = np.array(image)
        # print('Wounds: ', wounds)
        # print('Re-projected wounds: ', reprojected_wounds)
        for i in range(len(wounds)):
            # Draw original wound
            _image = cv2.circle(
                _image,
                (np.round(wounds[i][0]).astype('int'), np.round(wounds[i][1]).astype('int')),
                5,
                (255, 0, 0),
                2
            )
            # Draw reprojected wound
            try:
                if not (reprojected_wounds[i][0] == 0 and reprojected_wounds[i][1] == 0):
                    _image = cv2.circle(
                        _image,
                        (np.round(reprojected_wounds[i][0]).astype('int'),
                         np.round(reprojected_wounds[i][1]).astype('int')),
                        10,
                        (0, 0, 255), 2
                    )
            except:
                pass

        # Add "legend" to image
        _image = cv2.putText(_image, 'Original wound', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                             (255, 0, 0))
        _image = cv2.putText(_image, 'Re-projected wound', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                             (0, 0, 255))
        # Add total error as text on image
        _image = cv2.putText(_image, 'Total error: %.6f Pixels' % _error_total, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                             (255, 255, 255))

        if 'image_num' in kwargs:  # kwargs.get('image') contains the name of the image
            # Save images
            _filename = self.base_path + "/assets/images/" + str(kwargs.get('image_num')) + "_reprojected.jpg"
            cv2.imwrite(_filename, _image)

        if 'video_name' in kwargs:  # kwargs.get('video') contains the name of the video frame
            # Save reprojection images
            _filename = self.base_path + "/assets/videos/" + str(kwargs.get('video')) + "_reprojected.jpg"
            cv2.imwrite(_filename, _image)
