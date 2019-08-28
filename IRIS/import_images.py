#!/usr/bin/env python3
"""
This model is used to import the images and contain them into pixel-matrix.

We prepared 2 types of strategy to parse the different techniques of in situ sequencing, which published by R Ke,
CH Eng.

Here, Ke's data (R Ke, Nat. Methods, 2013) is employed as the major data type in our software, in this data, the
barcode are composed of 5 types of pseudo-color, representing the A, T, C, G bases and background.

In the type of Eng's data, Each image is composed of 4 channels, of which, the first 3 channels means blobs with
distinction by 3 pseudo-colors and the last one means background. Then, each continuous 4 images are made a Round, also
named a Cycle. So, there are 12 pseudo-colors in a Round. For example, the Eng's data (CH Eng, Nat. Methods, 2017)
include 5 Rounds, 20 images, 80 channels in any of shooting region.

Our software generate a 3D tensor to store all the images, each channel is made a image matrix, and insert into this
tensor in sequence of cycles
"""


from sys import stderr
from cv2 import (imread, imreadmulti, imwrite,
                 add, addWeighted, warpAffine,
                 IMREAD_GRAYSCALE,
                 error)
from numpy import (array,
                   uint8)

from ._register_images import register_cycles


def decode_data_Ke(f_cycles):
    """
    For parsing the technique which published on Nature Methods in 2013 by R Ke.

    Input the directories of cycle.
    Returning a pixel matrix which contains all the gray scales of image pixel as well as their coordinates.

    :param f_cycles: The image directories in sequence of cycles, of which the different channels are stored.
    :return: A tuple including a 3D common data tensor and a background image matrix.
    """
    if len(f_cycles) < 1:
        print('ERROR CYCLES', file=stderr)

        exit(1)

    f_cycle_stack = []

    f_std_img = array([], dtype=uint8)
    reg_ref = array([], dtype=uint8)

    for cycle_id in range(0, len(f_cycles)):
        adj_img_mats = []

        channel_A = imread('/'.join((f_cycles[cycle_id], 'Y5.tif')),   IMREAD_GRAYSCALE)
        channel_T = imread('/'.join((f_cycles[cycle_id], 'FAM.tif')),  IMREAD_GRAYSCALE)
        channel_C = imread('/'.join((f_cycles[cycle_id], 'TXR.tif')),  IMREAD_GRAYSCALE)
        channel_G = imread('/'.join((f_cycles[cycle_id], 'Y3.tif')),   IMREAD_GRAYSCALE)
        channel_0 = imread('/'.join((f_cycles[cycle_id], 'DAPI.tif')), IMREAD_GRAYSCALE)

        merged_img = addWeighted(add(add(add(channel_A, channel_T), channel_C), channel_G), 0.7, channel_0, 0.3, 0)

        if cycle_id == 0:
            reg_ref = channel_0
            f_std_img = merged_img

        try:
            trans_mat = register_cycles(reg_ref, channel_0, 'ORB')

        except error:
            trans_mat = register_cycles(reg_ref, channel_0, 'BRISK')

        imwrite('debug.cycle_' + str(int(cycle_id + 1)) + '.tif', merged_img)

        debug_img = warpAffine(merged_img, trans_mat, (reg_ref.shape[1], reg_ref.shape[0]))

        imwrite('debug.cycle_' + str(int(cycle_id + 1)) + '.reg.tif', debug_img)

        adj_img_mats.append(warpAffine(channel_A, trans_mat, (f_std_img.shape[1], f_std_img.shape[0])))
        adj_img_mats.append(warpAffine(channel_T, trans_mat, (f_std_img.shape[1], f_std_img.shape[0])))
        adj_img_mats.append(warpAffine(channel_C, trans_mat, (f_std_img.shape[1], f_std_img.shape[0])))
        adj_img_mats.append(warpAffine(channel_G, trans_mat, (f_std_img.shape[1], f_std_img.shape[0])))

        f_cycle_stack.append(adj_img_mats)

    return f_cycle_stack, f_std_img


def decode_data_Eng(f_cycles):
    """
    For parsing the technique which published on Nature Methods in 2013 and Nature in 2019 by CH Eng.

    Input the image files in each cycle.
    Returning a pixel matrix which contains all the gray scales of image pixel as well as their coordinates

    :param f_cycles: The image files in sequence of cycles. Each file include 4 channels.
    :return: A tuple including a 3D common data tensor and a background image matrix.
    """
    if len(f_cycles) % 4 != 0:
        print('ERROR ROUNDS', file=stderr)

        exit(1)

    f_cycle_stack = []

    f_std_img = array([], dtype=uint8)
    reg_ref = array([], dtype=uint8)

    for cycle_id in range(0, len(f_cycles), 4):
        adj_img_mats = []

        _, img_r1_mats = imreadmulti(f_cycles[cycle_id + 0], None, IMREAD_GRAYSCALE)
        _, img_r2_mats = imreadmulti(f_cycles[cycle_id + 1], None, IMREAD_GRAYSCALE)
        _, img_r3_mats = imreadmulti(f_cycles[cycle_id + 2], None, IMREAD_GRAYSCALE)
        _, img_r4_mats = imreadmulti(f_cycles[cycle_id + 3], None, IMREAD_GRAYSCALE)

        merged_img1 = img_r1_mats[3]
        merged_img2 = img_r2_mats[3]
        merged_img3 = img_r3_mats[3]
        merged_img4 = img_r4_mats[3]

        if cycle_id == 0:
            reg_ref = img_r1_mats[3]
            f_std_img = merged_img1

        try:
            trans_mat1 = register_cycles(reg_ref, img_r1_mats[3], 'ORB')
            trans_mat2 = register_cycles(reg_ref, img_r2_mats[3], 'ORB')
            trans_mat3 = register_cycles(reg_ref, img_r3_mats[3], 'ORB')
            trans_mat4 = register_cycles(reg_ref, img_r4_mats[3], 'ORB')

        except error:
            trans_mat1 = register_cycles(reg_ref, img_r1_mats[3], 'BRISK')
            trans_mat2 = register_cycles(reg_ref, img_r2_mats[3], 'BRISK')
            trans_mat3 = register_cycles(reg_ref, img_r3_mats[3], 'BRISK')
            trans_mat4 = register_cycles(reg_ref, img_r4_mats[3], 'BRISK')

        imwrite('debug.round_' + str(int(cycle_id / 4 + 1)) + '.tif', merged_img1)
        imwrite('debug.round_' + str(int(cycle_id / 4 + 1)) + '.tif', merged_img2)
        imwrite('debug.round_' + str(int(cycle_id / 4 + 1)) + '.tif', merged_img3)
        imwrite('debug.round_' + str(int(cycle_id / 4 + 1)) + '.tif', merged_img4)

        debug_img1 = warpAffine(merged_img1, trans_mat1, (reg_ref.shape[1], reg_ref.shape[0]))
        debug_img2 = warpAffine(merged_img2, trans_mat2, (reg_ref.shape[1], reg_ref.shape[0]))
        debug_img3 = warpAffine(merged_img3, trans_mat3, (reg_ref.shape[1], reg_ref.shape[0]))
        debug_img4 = warpAffine(merged_img4, trans_mat4, (reg_ref.shape[1], reg_ref.shape[0]))

        imwrite('debug.round_' + str(int(cycle_id / 4 + 1)) + '.reg.tif', debug_img1)
        imwrite('debug.round_' + str(int(cycle_id / 4 + 1)) + '.reg.tif', debug_img2)
        imwrite('debug.round_' + str(int(cycle_id / 4 + 1)) + '.reg.tif', debug_img3)
        imwrite('debug.round_' + str(int(cycle_id / 4 + 1)) + '.reg.tif', debug_img4)

        adj_img_mats.append(warpAffine(img_r1_mats[0], trans_mat1, (reg_ref.shape[1], reg_ref.shape[0])))
        adj_img_mats.append(warpAffine(img_r1_mats[1], trans_mat1, (reg_ref.shape[1], reg_ref.shape[0])))
        adj_img_mats.append(warpAffine(img_r1_mats[2], trans_mat1, (reg_ref.shape[1], reg_ref.shape[0])))

        adj_img_mats.append(warpAffine(img_r2_mats[0], trans_mat2, (reg_ref.shape[1], reg_ref.shape[0])))
        adj_img_mats.append(warpAffine(img_r2_mats[1], trans_mat2, (reg_ref.shape[1], reg_ref.shape[0])))
        adj_img_mats.append(warpAffine(img_r2_mats[2], trans_mat2, (reg_ref.shape[1], reg_ref.shape[0])))

        adj_img_mats.append(warpAffine(img_r3_mats[0], trans_mat3, (reg_ref.shape[1], reg_ref.shape[0])))
        adj_img_mats.append(warpAffine(img_r3_mats[1], trans_mat3, (reg_ref.shape[1], reg_ref.shape[0])))
        adj_img_mats.append(warpAffine(img_r3_mats[2], trans_mat3, (reg_ref.shape[1], reg_ref.shape[0])))

        adj_img_mats.append(warpAffine(img_r4_mats[0], trans_mat4, (reg_ref.shape[1], reg_ref.shape[0])))
        adj_img_mats.append(warpAffine(img_r4_mats[1], trans_mat4, (reg_ref.shape[1], reg_ref.shape[0])))
        adj_img_mats.append(warpAffine(img_r4_mats[2], trans_mat4, (reg_ref.shape[1], reg_ref.shape[0])))

        f_cycle_stack.append(adj_img_mats)

    return f_cycle_stack, f_std_img


# def decode_data_Weinstein(f_cycles):
#     """
#     For parsing the technique which published on Nature Methods in 2019 by JA Weinstein.
#
#     Input the directories of cycle.
#     Returning a pixel matrix which contains all the gray scales of image pixel as well as their coordinates
#
#     :param f_cycles:
#     :return:
#     """
#     pass


if __name__ == '__main__':
    pass
