# -*- coding: utf-8 -*-
__author__ = 'LiYuanhe'

import subprocess

import sys
import pathlib
Python_Lib_path = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(Python_Lib_path)
from My_Lib_Stock import *


def get_image_size(image_file):
    """
    read an image file, return (width, height) of the image file
    :param image_file:
    :return:
    """
    from PIL import Image
    return Image.open(image_file).size


def image_resize(image_file,
                 size: Sequence[int],
                 output_file=None):
    from PIL import Image
    image = Image.open(image_file)
    max_size = size
    aspect_ratio = image.width / image.height
    new_width = min(image.width, max_size[0])
    new_height = round(new_width / aspect_ratio)
    if new_height > max_size[1]:
        new_height = max_size[1]
        new_width = round(new_height * aspect_ratio)
    resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)
    if output_file is None:
        output_file = filename_class(image_file).insert_append('resize')
    resized_image.save(output_file)
    return output_file


if __name__ == '__main__':
    pass