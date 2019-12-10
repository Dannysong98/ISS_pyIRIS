#!/usr/bin/env python3


from sys import argv

from cv2 import (imread, imwrite, addWeighted,
                 IMREAD_GRAYSCALE)
from numpy import (array, uint8)


def merge_foreground_Ke(f_cycles, f_brightness=None):
    """"""
    brightness = float(f_brightness) if f_brightness is not None else 0.5
    foreground = array([], dtype=uint8)

    if len(f_cycles) == 1:

        for cycle_id in range(0, len(f_cycles)):
            channel_A = imread('/'.join((f_cycles[cycle_id], 'Y5.tif')),   IMREAD_GRAYSCALE)
            channel_T = imread('/'.join((f_cycles[cycle_id], 'FAM.tif')),  IMREAD_GRAYSCALE)
            channel_C = imread('/'.join((f_cycles[cycle_id], 'TXR.tif')),  IMREAD_GRAYSCALE)
            channel_G = imread('/'.join((f_cycles[cycle_id], 'Y3.tif')),   IMREAD_GRAYSCALE)

            foreground = addWeighted(addWeighted(addWeighted(channel_A, brightness,
                                                             channel_T, brightness, 0), brightness,
                                                 channel_C, brightness, 0), brightness,
                                     channel_G, brightness, 0)

    return foreground


if __name__ == '__main__':
    fg = merge_foreground_Ke(argv[1], argv[2])
    imwrite('background.new.tif', fg)
