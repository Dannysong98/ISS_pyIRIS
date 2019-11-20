#!/usr/bin/env python3


from sys import argv

from cv2 import (imread, IMREAD_GRAYSCALE)
from numpy import (array, sum)


def calculate_image_size(cycles):
    """"""
    tol_size = 0

    for cycle_id in range(0, len(cycles)):
        channel_A = imread('/'.join((cycles[cycle_id], 'Y5.tif')), IMREAD_GRAYSCALE)
        channel_T = imread('/'.join((cycles[cycle_id], 'FAM.tif')), IMREAD_GRAYSCALE)
        channel_C = imread('/'.join((cycles[cycle_id], 'TXR.tif')), IMREAD_GRAYSCALE)
        channel_G = imread('/'.join((cycles[cycle_id], 'Y3.tif')), IMREAD_GRAYSCALE)
        channel_0 = imread('/'.join((cycles[cycle_id], 'DAPI.tif')), IMREAD_GRAYSCALE)

        size = sum(array([channel_A.size, channel_T.size, channel_C.size, channel_G.size, channel_0.size]))

        tol_size += size

    return tol_size


if __name__ == '__main__':
    total_size = calculate_image_size(argv[1:])
    print(total_size)
