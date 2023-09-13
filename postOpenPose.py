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
    import wound_transformation_ros_node
except:
    print('Could not import rospy.')


def extract_keypoints(path):
    with open(path, 'r') as stream:
        data = yaml.safe_load(stream)
        try:
            assert not len(data['people']) == 0
        except:
            return {}
        keypoints_2d = data['people'][0]['pose_keypoints_2d']
        # Convert the list to a dictionary that consists of entries following (x,y,confidence)
        keypoints = {}
        j = 0
        for i in range(len(keypoints_2d)):
            if i % 3 == 0:
                keypoints[j] = {"x": keypoints_2d[i],
                                "y": keypoints_2d[i + 1],
                                "c": keypoints_2d[i + 2]}
                j = j + 1

        return keypoints


def handle_video(path, wound_transform, **kwargs):
    _video_path = path + '/assets/videos/' + kwargs.get('video_name')
    _video_name = kwargs.get('video_name').rsplit('.', 1)[0]
    _wound_path = path + '/assets/videos/wounds_' + _video_name + '.yaml'
    _keypoints_dir_path = path + '/assets/videos/keypoints_' + _video_name + '/'

    try:
        assert os.path.exists(_video_path)
        assert os.path.exists(_wound_path)
    except FileNotFoundError:
        print('Could not find the files in the specified path.')
        print('Video path: ', _video_path)
        print('Wound path: ', _wound_path)
        exit(-1)

    # Load wound data
    with open(_wound_path, 'r') as stream:
        _video_wounds = yaml.safe_load(stream)

    # Create VideoCapture object and load video
    _video_cap = cv2.VideoCapture(_video_path)

    # Define origin of frame counter on image
    org = (10, 30)

    # TEST!!!!!!
    # Heatmap for wound location, only works with a single wound!
    _standard_view = cv2.imread(path + '/assets/pictures/standard/male_100_rendered.png')
    _heatmap = np.zeros(_standard_view.shape)  # 348 x 612 is the shape of the standard view
    _error = []

    _out_path = path + '/assets/videos/' + _video_name + '_out.mp4'
    _fps = _video_cap.get(cv2.CAP_PROP_FPS)
    #_w = int(_video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #_h = int(_video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    _h, _w, _ = wound_transform.standard_view_img.shape
    _four_cc = cv2.VideoWriter_fourcc(*'mp4v')
    #print(_out_path, _four_cc, _fps, (_w, _h))
    _out = cv2.VideoWriter(_out_path, _four_cc, _fps, (_w, _h))
    # TEST!!!!!!

    # Read from _video_cap until video ends
    while _video_cap.isOpened():
        _ret, _frame = _video_cap.read()
        if _ret:
            _frame_num = int(_video_cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            _frame_name = _video_name + '_' + str(_frame_num).zfill(12)
            # Extract keypoints, this only works with openpose data!
            _keypoints = extract_keypoints(_keypoints_dir_path + _frame_name + '_keypoints.json')
            _wounds = [_video_wounds[_frame_num]]
            # print('_Wounds: ', _wounds)
            if len(_wounds[0]) > 0:
                try:
                    # Step 0: Display wound on original video-frame and write to video
                    #_image_before = cv2.circle(np.ndarray.copy(_frame), _wounds[0], 5, (0, 0, 255), 2)
                    #_out.write(_image_before)
                    # Step 1: Convert wounds from random pose on image to standard view reference frame
                    _conv_wounds = wound_transform.locate_wounds(_frame, _keypoints, _wounds)
                    if _conv_wounds[0][0] == -1 and _conv_wounds[0][1] == -1:
                        _heatmap[_conv_wounds[0][1], _conv_wounds[0][0]] += 0
                    else:
                        _heatmap[_conv_wounds[0][1], _conv_wounds[0][0]] += 1
                    # Step 2: Transfer converted wounds onto the standard view
                    _img_conv = wound_transform.transform_view(_conv_wounds)
                    # Step 3: Re-project converted wounds onto original image
                    _reprojected_wounds = wound_transform.reproject_wounds(_keypoints, _wounds, _conv_wounds)
                    # Step 4: Calculate reprojected error
                    _error.append(wound_transformation.calculate_reprojection_error(_wounds, _reprojected_wounds))
                    # Step 5: Write image of transformed view to video
                    _out.write(_img_conv)
                    #wound_transform.display_error(_frame, _wounds, _reprojected_wounds, video=_frame_name)
                except:
                    _error.append([-1])
                    #_standard_view_no_detect = cv2.putText(np.ndarray.copy(_standard_view), 'Could not transfer wound.',
                    #                                       org, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    _out.write(_standard_view)
            else:
                _standard_view_no_wound = cv2.putText(np.ndarray.copy(_standard_view), 'No wound found.', org,
                                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                _out.write(_standard_view_no_wound)
                _error.append([0])
            # cv2.imshow('frame', _frame)
            # cv2.waitKey(10)

        else:
            break

    _error_per_wound = np.array(_error).flatten()
    with open(path + '/assets/videos/error_per_wound_' + _video_name + '.yaml', 'w') as error_file:
        yaml.dump(_error_per_wound.tolist(), error_file)
    wound_transform.plot_error(_error_per_wound, video=_video_name)

    _ksize = (3, 3)   # Kernel size
    _sigma = 10  # Standard deviation of the Gaussian distribution
    _heatmap_extended = cv2.GaussianBlur(_heatmap, _ksize, _sigma)
    _heatmap_normalized = _heatmap_extended / np.max(_heatmap_extended)
    _heatmap_colored = cv2.applyColorMap(np.uint8(255 * _heatmap_extended), cv2.COLORMAP_JET)
    _standard_view = cv2.imread(path + '/assets/pictures/standard/male_100_rendered.png')

    #print(_heatmap_colored.shape)
    #print(_standard_view.shape)
    #cv2.imshow('heatmap', _heatmap_colored)
    #cv2.waitKey(0)

    _applied_heatmap = cv2.addWeighted(_standard_view, 0.7, _heatmap_colored, 0.3, 0)

    cv2.imwrite(path + '/assets/videos/' + _video_name + '_converted_wound_heatmap.jpg', _applied_heatmap)
    #wound_transform.plot_error(_video_wounds)

    #cv2.imshow('Resultat', _applied_heatmap)
    #cv2.waitKey(0)


def main(**kwargs):
    if 'host' in kwargs and kwargs.get('host') == 'local':
        _base_path = os.path.dirname(
            os.path.abspath("postOpenPose.py")
        )
        if 'skeleton' in kwargs and kwargs.get('skeleton') == 'nuitrack':
            wound_transform = wound_transformation.WoundTransformation(skeleton=kwargs.get('nuitrack'))
        else:
            wound_transform = wound_transformation.WoundTransformation(skeleton=kwargs.get('skeleton'))
        # Analyze and transform wounds live or from saved images?
        if 'mode' in kwargs and kwargs.get('mode') == 'video':
            # Get video path
            if 'video_name' in kwargs:
                handle_video(_base_path, wound_transform, video_name=kwargs.get('video_name'))

        elif 'mode' in kwargs and kwargs.get('mode') == 'image':
            if 'image_num' in kwargs:
                _num = kwargs['image_num']
            else:
                try:
                    # No number was given, choose a random number between 0 and the amount of pictures in the rs_color
                    # folder
                    _num_img = len(os.listdir(_base_path + '/assets/pictures/rs_color/'))
                    _num = np.random.randint(0, _num_img)
                except:
                    print('Could not find path, please run the script preOpenPose.py '
                          'to calibrate the cameras and create pictures')
                    exit(1)
            # Load image
            _img = cv2.imread(_base_path + '/assets/pictures/rs_color/' + str(_num) + '.jpg')
            try:
                _img_d = cv2.imread(_base_path + '/assets/pictures/rs_depth/' + str(_num) + '.jpg')
            except:
                _img_d = None
            # Load keypoints
            if 'skeleton' in kwargs and kwargs.get('skeleton') == 'nui':
                print('NuiTrack not yet implemented.')
                exit(1)
            else:
                _keypoints = extract_keypoints(_base_path + '/assets/pictures/keypoints/' + str(_num) + '_keypoints.json')
            # Load the wounds data
            with open(_base_path + '/assets/pictures/wounds/' + str(_num) + '.yaml', 'r') as stream:
                _wounds = yaml.safe_load(stream)

            # _img, _wounds, _keypoints are loaded, proceed with wound transformation
            #print(_wounds)
            _img_copy = np.copy(_img)
            for wound in _wounds:
                _img_copy = cv2.circle(_img_copy, (np.round(wound[0]).astype('int'), np.round(wound[1]).astype('int')),
                                       5, (0, 0, 255), 2)
            # Step 1: Convert wounds from random pose on image to standard view reference frame
            _conv_wounds = wound_transform.locate_wounds(_img, _keypoints, _wounds, _img_d, _num)
            print('Converted Wounds: ', _conv_wounds)
            # Step 2: Transfer converted wounds onto the standard view
            _img_conv = wound_transform.transform_view(_conv_wounds)
            # Step 2.5: Save image to output
            #wound_transform.save_images(after=_img_conv, num=_num)
            # Step 3: Re-project converted wounds onto original image
            #_reprojected_wounds = wound_transform.reproject_wounds(_keypoints, _wounds, _conv_wounds)
            # Step 4: Calculate and display reprojected error
            #wound_transform.display_error(_img, _wounds, _reprojected_wounds, image_num=_num)
            #error_per_wound = wound_transformation.calculate_reprojection_error(_wounds, _reprojected_wounds)
            #wound_transform.plot_error(error_per_wound, image=_num)
    else:
        # No local data acquisition, use ROS
        wound_transformation_ros_node.WoundTransformROSNode()
        rospy.spin()


if __name__ == '__main__':
    # Possible optional parameters:
    # host: local           If host='local' provided, non-ROS version will be started
    # mode: image/video     If any mode is provided and non-ROS version started, either an image or a video will be
    #                       processed
    # image_num: int        If any image_num is provided and non-ROS, image version started, the image with the
    #                       specified image_num will be used
    # video_name: String    If any video_name is provided and non-ROS, video version was started, the video with the
    #                       specified name will be used
    """
    Example for image mode using openpose:
    main(mode='image', num=9, skeleton='openpose', host='local')
    
    Example for video variant: (Not yet implemented)
    main(mode='video', video_name=sampleVid.mp4, host='local')
    
    Example for ROS:
    main()
    """
    main()
