# This script is part of the bachelor thesis
# "Localization of medical measurement data from injured persons for use in rescue robotics"
# from Konstantin Wenig written for the university to luebeck's bachelor degree in robotics and autonomous systems
# Redistribution and change without consent it not allowed

import numpy as np
try:
    import pyrealsense2 as rs
except:
    print("Could not find pyrealsense, only offline data available!")


class RealsenseCam:
    def __init__(self):

        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Get device product line for setting a supporting resolution
        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
        self.device = self.pipeline_profile.get_device()
        self.device_product_line = str(self.device.get_info(rs.camera_info.product_line))

        # In case the script ever uses a different camera, check if the current
        # camera is sufficient for this script
        self.found_rgb = False
        for s in self.device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                self.found_rgb = True
                break
        if not self.found_rgb:
            print("This script requires a Depth camera with Color sensor")
            exit(0)

        # Enable stream for depth
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # Enable stream for color
        if self.device_product_line == 'L500':
            self.config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        self.pipeline.start(self.config)

    def get_frame(self):
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if depth_frame and color_frame:
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data()).copy()
            color_image = np.asanyarray(color_frame.get_data()).copy()
            return color_image, depth_image

    def get_frame_size(self):
        frame_x, frame_y = np.asanyarray(self.pipeline.wait_for_frames().get_color_frame().get_data()).shape
        return frame_x, frame_y

    def stop_pipeline(self):
        self.pipeline.stop()
