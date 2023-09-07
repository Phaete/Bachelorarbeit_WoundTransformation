# This script is part of the bachelor thesis
# "Localization of medical measurement data from injured persons for use in rescue robotics"
# from Konstantin Wenig written for the university to luebeck's bachelor degree in robotics and autonomous systems

# The content of this script is extracted from the script named "uvc-radiometry.py" from https://github.com/groupgets/purethermal1-uvc-capture
"""
try:
    from uvctypes import *
except:
    print('Could not find libuvc!')
import cv2
import numpy as np

try:
    from queue import Queue
except ImportError:
    from queue import Queue

q = Queue(2)


# Convert 16-bit to 8-bit
def raw_to_8bit(data):
    img = cv2.normalize(data, None, 0, 256, cv2.NORM_MINMAX)
    return cv2.cvtColor(np.uint8(img), cv2.COLOR_GRAY2RGB)


def py_frame_callback(frame, _):
    try:
        global q
        array_pointer = cast(frame.contents.data, POINTER(c_uint16 * (frame.contents.width * frame.contents.height)))
        data = np.frombuffer(
                array_pointer.contents, dtype=np.dtype(np.uint16)
                ).reshape(
                frame.contents.height, frame.contents.width
            )

        if frame.contents.data_bytes != (2 * frame.contents.width * frame.contents.height):
            return

        q.put(data)
    except NameError:
        print('libuvc not found, could not prepare py_frame_callback')

"""
class ThermalCam:
    pass
    """
    try:
        _PTR_PY_FRAME_CALLBACK = CFUNCTYPE(None, POINTER(uvc_frame), c_void_p)(py_frame_callback)
    except NameError:
        print('Could not properly initiate thermalcam object')

    def __init__(self):
        global q
        self.q = q

        # Code extracted from source 1 to create the uvc device
        self.ctx = POINTER(uvc_context)()
        self.dev = POINTER(uvc_device)()
        self.devh = POINTER(uvc_device_handle)()
        self.ctrl = uvc_stream_ctrl()

        self.res = libuvc.uvc_init(byref(self.ctx), 0)
        if self.res < 0:
            print("uvc_init error")
            exit(1)

        try:
            self.res = libuvc.uvc_find_device(self.ctx, byref(self.dev), PT_USB_VID, PT_USB_PID, 0)
            if self.res < 0:
                print("uvc_find_device error")
                exit(1)

            try:
                self.res = libuvc.uvc_open(self.dev, byref(self.devh))
                if self.res < 0:
                    print("uvc_open error")
                    exit(1)

                print("device opened!")

                print_device_info(self.devh)
                print_device_formats(self.devh)

                self.frame_formats = uvc_get_frame_formats_by_guid(self.devh, VS_FMT_GUID_Y16)
                if len(self.frame_formats) == 0:
                    print("device does not support Y16")
                    exit(1)

                libuvc.uvc_get_stream_ctrl_format_size(self.devh, byref(self.ctrl), UVC_FRAME_FORMAT_Y16,
                                                       self.frame_formats[0].wWidth, self.frame_formats[0].wHeight,
                                                       int(1e7 / self.frame_formats[0].dwDefaultFrameInterval)
                                                       )

                self.res = libuvc.uvc_start_streaming(self.devh, byref(self.ctrl), ThermalCam._PTR_PY_FRAME_CALLBACK,
                                                      None, 0)
                if self.res < 0:
                    print("uvc_start_streaming failed: {0}".format(self.res))
                    exit(1)

            except:
                print("uvc_open error")
                libuvc.uvc_unref_device(self.dev)
        except:
            print("uvc_find_device error")
            libuvc.uvc_exit(self.ctx)

    def get_frame(self):
        data = self.q.get(True, 50)
        if data is None:
            return -1
        else:
            return data[:, 10:70]

    def stop_stream(self):
        libuvc.uvc_stop_streaming(self.devh)
"""

