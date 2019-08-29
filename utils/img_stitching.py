#!/usr/bin/env python3


from sys import (argv, exit,
                 stderr)
from cv2 import (imread, createStitcherScans,
                 IMREAD_GRAYSCALE)

from ..IRIS.register_images import register_cycles


def background_stitcher(paths):
    """"""
    imgs = []

    for img_path in paths:
        img = imread(img_path, IMREAD_GRAYSCALE)
        imgs.append(img)

    stitcher = createStitcherScans()
    _, stitched_img = stitcher.stitch(imgs)

    if _ == 0:
        return stitched_img

    else:
        print('FAIL TO STITCH', file=stderr)
        exit(1)


def get_matrix(bg, imgs):
    """"""
    mats = []
    for img in imgs:
        mats.append(register_cycles(bg, img, 'BRISK'))
