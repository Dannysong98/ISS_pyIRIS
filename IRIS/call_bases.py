#!/usr/bin/env python3
"""
This module is used to transform detected fluorescence signals into barcode sequence.

Each cycle is composed of at several channels, such as A, T, C and G in Ke's data structure. Sometimes, an additional
in situ hybridization signal (DO) and nucleus staining signal (DAPI) are also provided.

We select the channel with the highest base score against the other channels at a same location as the representative
channel in a certain location. Base quality is calculated as follows: we calcuclate p-value via a binomial test by
taking the highest base score as the number of success and the second highest base score as the number of failure and
treat it as error rate.
"""


from numpy import (around, transpose, nonzero)
from scipy.stats import binom_test


def image_model_pooling_Ke(f_image_model_A, f_image_model_T, f_image_model_C, f_image_model_G):
    """
    :param f_image_model_A: Channel A in a cycle from the 3D data matrix.
    :param f_image_model_T: Channel T in a cycle from the 3D data matrix.
    :param f_image_model_C: Channel C in a cycle from the 3D data matrix.
    :param f_image_model_G: Channel G in a cycle from the 3D data matrix.
    :return f_image_model_pool: A dictionary of blobs with its base, location and base score
    """
    f_image_model_pool = {}

    #################################################
    # Collect all the coordinates of detected blobs #
    #################################################
    f_image_model = f_image_model_A + f_image_model_T + f_image_model_C + f_image_model_G
    #################################################

    ##############################################################################################################
    # Each coordinate stores the base scores, and the largest one is made to be the representative of this cycle #
    # the second highest base score will also be used to calculate base quality                                  #
    ##############################################################################################################
    for row, col in transpose(nonzero(f_image_model)):
        #######################################################################################################
        # Our software could handle the images no larger than 99999x99999                                     #
        # This size limit should fit most of images                                                           #
        # You can modify this limit like following options in each place of 'read_id' for fitting your images #
        #######################################################################################################
        read_id = 'r%05dc%05d' % (row + 1, col + 1)
        ########
        # read_id = 'r%06dc%06d' % (row + 1, col + 1)  # Alternative option
        # read_id = 'r%07dc%07d' % (row + 1, col + 1)  # Alternative option
        # read_id = 'r%08dc%08d' % (row + 1, col + 1)  # Alternative option

        #######################################################################################################

        if read_id not in f_image_model_pool:
            f_image_model_pool.update({read_id: {'A': 0, 'T': 0, 'C': 0, 'G': 0}})

        if f_image_model_A[row, col] > 0:
            f_image_model_pool[read_id]['A'] = f_image_model_A[row, col]

        if f_image_model_T[row, col] > 0:
            f_image_model_pool[read_id]['T'] = f_image_model_T[row, col]

        if f_image_model_C[row, col] > 0:
            f_image_model_pool[read_id]['C'] = f_image_model_C[row, col]

        if f_image_model_G[row, col] > 0:
            f_image_model_pool[read_id]['G'] = f_image_model_G[row, col]
    ##############################################################################################################

    return f_image_model_pool


def image_model_pooling_Eng(f_image_model_1, f_image_model_2, f_image_model_3,
                            f_image_model_4, f_image_model_5, f_image_model_6,
                            f_image_model_7, f_image_model_8, f_image_model_9,
                            f_image_model_A, f_image_model_B, f_image_model_C):
    """
    :param f_image_model_1: Channel 1 in a cycle from the 3D data matrix.
    :param f_image_model_2: Channel 2 in a cycle from the 3D data matrix.
    :param f_image_model_3: Channel 3 in a cycle from the 3D data matrix.
    :param f_image_model_4: Channel 4 in a cycle from the 3D data matrix.
    :param f_image_model_5: Channel 5 in a cycle from the 3D data matrix.
    :param f_image_model_6: Channel 6 in a cycle from the 3D data matrix.
    :param f_image_model_7: Channel 7 in a cycle from the 3D data matrix.
    :param f_image_model_8: Channel 8 in a cycle from the 3D data matrix.
    :param f_image_model_9: Channel 9 in a cycle from the 3D data matrix.
    :param f_image_model_A: Channel A in a cycle from the 3D data matrix.
    :param f_image_model_B: Channel B in a cycle from the 3D data matrix.
    :param f_image_model_C: Channel C in a cycle from the 3D data matrix.
    :return f_image_model_pool: A dictionary of blobs with its base, location and base score.
    """
    f_image_model_pool = {}

    f_image_model = f_image_model_1 + f_image_model_2 + f_image_model_3 + \
                    f_image_model_4 + f_image_model_5 + f_image_model_6 + \
                    f_image_model_7 + f_image_model_8 + f_image_model_9 + \
                    f_image_model_A + f_image_model_B + f_image_model_C

    for row, col in transpose(nonzero(f_image_model)):
        read_id = 'r%05dc%05d' % (row + 1, col + 1)

        if read_id not in f_image_model_pool:
            f_image_model_pool.update({read_id: {'1': 0, '2': 0, '3': 0,
                                                 '4': 0, '5': 0, '6': 0,
                                                 '7': 0, '8': 0, '9': 0,
                                                 'A': 0, 'B': 0, 'C': 0}})

        if f_image_model_1[row, col] > 0:
            f_image_model_pool[read_id]['1'] = f_image_model_1[row, col]

        if f_image_model_2[row, col] > 0:
            f_image_model_pool[read_id]['2'] = f_image_model_2[row, col]

        if f_image_model_3[row, col] > 0:
            f_image_model_pool[read_id]['3'] = f_image_model_3[row, col]

        if f_image_model_4[row, col] > 0:
            f_image_model_pool[read_id]['4'] = f_image_model_4[row, col]

        if f_image_model_5[row, col] > 0:
            f_image_model_pool[read_id]['5'] = f_image_model_5[row, col]

        if f_image_model_6[row, col] > 0:
            f_image_model_pool[read_id]['6'] = f_image_model_6[row, col]

        if f_image_model_7[row, col] > 0:
            f_image_model_pool[read_id]['7'] = f_image_model_7[row, col]

        if f_image_model_8[row, col] > 0:
            f_image_model_pool[read_id]['8'] = f_image_model_8[row, col]

        if f_image_model_9[row, col] > 0:
            f_image_model_pool[read_id]['9'] = f_image_model_9[row, col]

        if f_image_model_A[row, col] > 0:
            f_image_model_pool[read_id]['A'] = f_image_model_A[row, col]

        if f_image_model_B[row, col] > 0:
            f_image_model_pool[read_id]['B'] = f_image_model_B[row, col]

        if f_image_model_C[row, col] > 0:
            f_image_model_pool[read_id]['C'] = f_image_model_C[row, col]

    return f_image_model_pool


def image_model_pooling_Chen(f_image_model_0):
    """
    :param f_image_model_0: Channel 0 in a cycle from the 3D data matrix.
    :return f_image_model_pool: A dictionary of blobs with its base, location and base score
    """
    f_image_model_pool = {}

    #################################################
    # Collect all the coordinates of detected blobs #
    #################################################
    f_image_model = f_image_model_0
    #################################################

    ##############################################################################################################
    # Each coordinate stores the base scores, and the largest one is made to be the representative of this cycle #
    # the second highest base score will also be used to calculate base quality                                  #
    ##############################################################################################################
    for row, col in transpose(nonzero(f_image_model)):
        #######################################################################################################
        # Our software could handle the images no larger than 99999x99999                                     #
        # This size limit should fit most of images                                                           #
        # You can modify this limit like following options in each place of 'read_id' for fitting your images #
        #######################################################################################################
        read_id = 'r%05dc%05d' % (row + 1, col + 1)
        ########
        # read_id = 'r%06dc%06d' % (row + 1, col + 1)  # Alternative option
        # read_id = 'r%07dc%07d' % (row + 1, col + 1)  # Alternative option
        # read_id = 'r%08dc%08d' % (row + 1, col + 1)  # Alternative option

        #######################################################################################################

        if read_id not in f_image_model_pool:
            f_image_model_pool.update({read_id: {'S': 0}})

        if f_image_model_0[row, col] > 0:
            f_image_model_pool[read_id]['S'] = f_image_model_0[row, col]
    ##############################################################################################################

    return f_image_model_pool


def pool2base(f_image_model_pool):
    """
    :param f_image_model_pool: The dictionary of blobs, including base, coordinate and base score.
    :return f_base_box: A dictionary of blobs with its base, location and base error rate.
    """
    f_base_box = {}

    for read_id in f_image_model_pool:
        sorted_base = [_ for _ in sorted(f_image_model_pool[read_id].items(), key=lambda x: x[1], reverse=True)]

        if sorted_base[0][1] > sorted_base[1][1]:
            error_rate = around(binom_test((sorted_base[0][1], sorted_base[1][1]), p=0.5, alternative='greater'), 4)

            if read_id not in f_base_box:
                f_base_box.update({read_id: [sorted_base[0][0], error_rate]})

    return f_base_box


def pool2base2(f_image_model_pool):
    """
    :param f_image_model_pool: The dictionary of blobs, including base, coordinate and base score.
    :return f_base_box: A dictionary of blobs with its base, location.
    """
    f_base_box = {}

    for read_id in f_image_model_pool:
        sorted_base = [_ for _ in sorted(f_image_model_pool[read_id].items(), key=lambda x: x[1], reverse=True)]

        if read_id not in f_base_box:
            f_base_box.update({read_id: [sorted_base[0][0], sorted_base[0][1]]})

    return f_base_box


if __name__ == '__main__':
    pass
