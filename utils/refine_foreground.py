#!/usr/bin/env python3


from sys import argv

from cv2 import (imread, imwrite, addWeighted,
                 IMREAD_GRAYSCALE)
from numpy import (array, uint8)


def merge_foreground_Ke(f_cycles, f_brightness1=None, f_brightness2=None):
    """"""
    brightness1 = float(f_brightness1) if f_brightness1 is not None else 0.5
    brightness2 = float(f_brightness2) if f_brightness2 is not None else 0.5
    foreground = array([], dtype=uint8)

    if len(f_cycles) == 1:

        for cycle_id in range(0, len(f_cycles)):
            channel_A = imread('/'.join((f_cycles[cycle_id], 'Y5.tif')),   IMREAD_GRAYSCALE)
            channel_T = imread('/'.join((f_cycles[cycle_id], 'FAM.tif')),  IMREAD_GRAYSCALE)
            channel_C = imread('/'.join((f_cycles[cycle_id], 'TXR.tif')),  IMREAD_GRAYSCALE)
            channel_G = imread('/'.join((f_cycles[cycle_id], 'Y3.tif')),   IMREAD_GRAYSCALE)
            channel_0 = imread('/'.join((f_cycles[cycle_id], 'DAPI.tif')),   IMREAD_GRAYSCALE)

            foreground = addWeighted(
                addWeighted(
                    addWeighted(
                        addWeighted(channel_A, 0.5,
                                    channel_T, 0.5, 0), 0.5,
                        channel_C, 0.5, 0), 0.5,
                    channel_G, 0.5, 0), brightness1,
                channel_0, brightness2, 0)

    return foreground


if __name__ == '__main__':
    try:
        fg = merge_foreground_Ke(argv[1], argv[2], argv[3])

    except IndexError:
        fg = merge_foreground_Ke(argv[1])

    imwrite('background.new.tif', fg)
