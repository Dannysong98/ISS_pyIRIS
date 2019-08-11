#!/usr/bin/env python3
"""
This model is used to import the images and contain them into pixel-matrix.

We prepared 2 types of strategy to parse the different techniques of in situ sequencing, which published by R. Ke,
CH. Eng.

Here, Ke's data is employed as the major data type in our software, in this data, the barcode are composed of 5 types of
pseudo-color, representing the A, T, C, G bases and background. While, in the type of Eng' data, Each image is composed
of 4 channels, of which, the first 3 channels means blobs with distinction by 3 pseudo-colors and the last one means
background. Then, each continuous 4 images are made a Round, also named a Cycle. So, there are 12 pseudo-colors in
a Round. The Eng's data (CH. Eng, Nat Met., 2017) include 5 Rounds, 20 images, 80 channels.

Our software generate a 3D tensor to store all the images, each channel is made into a matrix.
"""


from sys import stderr

from cv2.cv2 import (imread, imreadmulti,
                     add, addWeighted, warpAffine,
                     IMREAD_GRAYSCALE)

from numpy import (array,
                   uint8)

from ._register_images import register_cycles


def decode_data_Ke(f_cycles):
    """
    For parsing the technique which published on Nature Methods in 2013 by R. Ke.

    Input the directories of cycle.
    Returning a pixel matrix which contains all the gray scales of image pixel as well as their coordinates.
    """
    if len(f_cycles) < 1:
        print('ERROR CYCLES', file=stderr)

        exit(1)

    f_cycle_stack = []

    f_std_cycle = array([], dtype=uint8)

    for cycle_id in range(0, len(f_cycles)):
        channel_A = imread('/'.join((f_cycles[cycle_id], 'Y5.tif')),   IMREAD_GRAYSCALE)
        channel_T = imread('/'.join((f_cycles[cycle_id], 'FAM.tif')),  IMREAD_GRAYSCALE)
        channel_C = imread('/'.join((f_cycles[cycle_id], 'TXR.tif')),  IMREAD_GRAYSCALE)
        channel_G = imread('/'.join((f_cycles[cycle_id], 'Y3.tif')),   IMREAD_GRAYSCALE)
        channel_0 = imread('/'.join((f_cycles[cycle_id], 'DAPI.tif')), IMREAD_GRAYSCALE)

        merge_cycle = addWeighted(add(add(add(channel_A, channel_T), channel_C), channel_G), 0.7, channel_0, 0.3, 0)

        if cycle_id == 0:
            f_std_cycle = merge_cycle

        trans_matrix = register_cycles(f_std_cycle, merge_cycle, 'BRISK')

        adj_channel_A = warpAffine(channel_A, trans_matrix, (f_std_cycle.shape[1], f_std_cycle.shape[0]))
        adj_channel_T = warpAffine(channel_T, trans_matrix, (f_std_cycle.shape[1], f_std_cycle.shape[0]))
        adj_channel_C = warpAffine(channel_C, trans_matrix, (f_std_cycle.shape[1], f_std_cycle.shape[0]))
        adj_channel_G = warpAffine(channel_G, trans_matrix, (f_std_cycle.shape[1], f_std_cycle.shape[0]))
        adj_channel_0 = warpAffine(channel_0, trans_matrix, (f_std_cycle.shape[1], f_std_cycle.shape[0]))

        f_cycle_stack.append((adj_channel_A, adj_channel_T, adj_channel_C, adj_channel_G, adj_channel_0))

    return f_cycle_stack, f_std_cycle


def decode_data_Eng(f_cycles):
    """
    For parsing the technique which published on Nature Methods in 2013 and Nature in 2019 by Chee-Huat Linus Eng.

    Input the directories of cycle.
    Returning a pixel matrix which contains all the gray scales of image pixel as well as their coordinates
    """
    if len(f_cycles) % 4 != 0:
        print('ERROR ROUNDS', file=stderr)

        exit(1)

    f_cycle_stack = []

    f_std_cycle = array([], dtype=uint8)

    for cycle_id in range(0, len(f_cycles), 4):
        adj_img_r1_mats = []
        adj_img_r2_mats = []
        adj_img_r3_mats = []
        adj_img_r4_mats = []

        _, f_img_r1_mats = imreadmulti(f_cycles[cycle_id + 0], None, IMREAD_GRAYSCALE)
        _, f_img_r2_mats = imreadmulti(f_cycles[cycle_id + 1], None, IMREAD_GRAYSCALE)
        _, f_img_r3_mats = imreadmulti(f_cycles[cycle_id + 2], None, IMREAD_GRAYSCALE)
        _, f_img_r4_mats = imreadmulti(f_cycles[cycle_id + 3], None, IMREAD_GRAYSCALE)

        if cycle_id == 0:
            f_std_cycle = f_img_r1_mats[3]

        trans_mat1 = register_cycles(f_std_cycle, f_img_r1_mats[3], 'BRISK')
        trans_mat2 = register_cycles(f_std_cycle, f_img_r2_mats[3], 'BRISK')
        trans_mat3 = register_cycles(f_std_cycle, f_img_r3_mats[3], 'BRISK')
        trans_mat4 = register_cycles(f_std_cycle, f_img_r4_mats[3], 'BRISK')

        adj_img_r1_mats.append(warpAffine(f_img_r1_mats[0], trans_mat1, (f_std_cycle.shape[1], f_std_cycle.shape[0])))
        adj_img_r1_mats.append(warpAffine(f_img_r1_mats[1], trans_mat1, (f_std_cycle.shape[1], f_std_cycle.shape[0])))
        adj_img_r1_mats.append(warpAffine(f_img_r1_mats[2], trans_mat1, (f_std_cycle.shape[1], f_std_cycle.shape[0])))
        adj_img_r1_mats.append(warpAffine(f_img_r1_mats[3], trans_mat1, (f_std_cycle.shape[1], f_std_cycle.shape[0])))

        adj_img_r2_mats.append(warpAffine(f_img_r2_mats[0], trans_mat2, (f_std_cycle.shape[1], f_std_cycle.shape[0])))
        adj_img_r2_mats.append(warpAffine(f_img_r2_mats[1], trans_mat2, (f_std_cycle.shape[1], f_std_cycle.shape[0])))
        adj_img_r2_mats.append(warpAffine(f_img_r2_mats[2], trans_mat2, (f_std_cycle.shape[1], f_std_cycle.shape[0])))
        adj_img_r2_mats.append(warpAffine(f_img_r2_mats[3], trans_mat2, (f_std_cycle.shape[1], f_std_cycle.shape[0])))

        adj_img_r3_mats.append(warpAffine(f_img_r3_mats[0], trans_mat3, (f_std_cycle.shape[1], f_std_cycle.shape[0])))
        adj_img_r3_mats.append(warpAffine(f_img_r3_mats[1], trans_mat3, (f_std_cycle.shape[1], f_std_cycle.shape[0])))
        adj_img_r3_mats.append(warpAffine(f_img_r3_mats[2], trans_mat3, (f_std_cycle.shape[1], f_std_cycle.shape[0])))
        adj_img_r3_mats.append(warpAffine(f_img_r3_mats[3], trans_mat3, (f_std_cycle.shape[1], f_std_cycle.shape[0])))

        adj_img_r4_mats.append(warpAffine(f_img_r4_mats[0], trans_mat4, (f_std_cycle.shape[1], f_std_cycle.shape[0])))
        adj_img_r4_mats.append(warpAffine(f_img_r4_mats[1], trans_mat4, (f_std_cycle.shape[1], f_std_cycle.shape[0])))
        adj_img_r4_mats.append(warpAffine(f_img_r4_mats[2], trans_mat4, (f_std_cycle.shape[1], f_std_cycle.shape[0])))
        adj_img_r4_mats.append(warpAffine(f_img_r4_mats[3], trans_mat4, (f_std_cycle.shape[1], f_std_cycle.shape[0])))

        f_cycle_stack.append((adj_img_r1_mats, adj_img_r2_mats, adj_img_r3_mats, adj_img_r4_mats))

    return f_cycle_stack, f_std_cycle


# def decode_data_Weinstein(f_cycles):
#     """
#     For parsing the technique which published on Nature Methods in 2019 by Joshua A. Weinstein.
#
#     Input the directories of cycle.
#     Returning a pixel matrix which contains all the gray scales of image pixel as well as their coordinates
#     """
#     pass


if __name__ == '__main__':
    pass
