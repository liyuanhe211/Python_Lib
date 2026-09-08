# -*- coding: utf-8 -*-
__author__ = 'LiYuanhe'

from Python_Lib.My_Lib_Stock import *


def video_duration_s(filename):
    import cv2
    video = cv2.VideoCapture(filename)

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    # print(filename,fps,frame_count)
    assert fps, filename
    return frame_count / fps


if __name__ == '__main__':
    pass
