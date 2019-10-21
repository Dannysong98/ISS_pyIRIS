#!/usr/bin/env python3


from sys import argv
from cv2 import (imread, imwrite, add,
                 IMREAD_GRAYSCALE)
from numpy import (array, uint8)


def merge_foreground_Ke(f_cycles):
    """"""
    foreground = array([], dtype=uint8)

    if len(f_cycles) == 1:

        for cycle_id in range(0, len(f_cycles)):
            channel_A = imread('/'.join((f_cycles[cycle_id], 'Y5.tif')),   IMREAD_GRAYSCALE)
            channel_T = imread('/'.join((f_cycles[cycle_id], 'FAM.tif')),  IMREAD_GRAYSCALE)
            channel_C = imread('/'.join((f_cycles[cycle_id], 'TXR.tif')),  IMREAD_GRAYSCALE)
            channel_G = imread('/'.join((f_cycles[cycle_id], 'Y3.tif')),   IMREAD_GRAYSCALE)

            foreground = add(add(add(channel_A, channel_T), channel_C), channel_G)

    return foreground


if __name__ == '__main__':
    fg = merge_foreground_Ke(argv[1])
    imwrite('background.new.tif', fg)
