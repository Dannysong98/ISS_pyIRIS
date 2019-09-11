#!/usr/bin/env python3
"""
This class is used to transform the detected fluorescence signal into sequence information.

Each cycle is composed of at least 4 channels, A, T, C and G. Sometimes, an additional in situ hybridization signal
(DO) and nucleus staining signal (DAPI) are also provided.

In our practice, we select the one taking the highest blob signal score against other 3 channels from a same
location as the representation in a certain location. And its sequencing quality is calculated by a novel method, in
this method, we presume that the difference between the base taking highest blob signal score and the one taking
second higher blob signal score is positive correlation with reliability, and is no correlation with other 2 blob
signal scores. It means the probability of error happen should approximately appear binomial distribution. We employ
binomial test to evaluate the difference between these top two blob signal score channels, and record the p-value as
the error rate to calculate sequence quality.

Besides, the coordinate of many detected blobs group which should be in a same location in pixel level among
different cycles are different, because of the error of registration. If a difference are light enough, we still
consider they located in a same location. Here we use pyramid shadow from a blob in cycle 1 to search 8x8 region in
other cycle, and find the blob taking highest gray scale as the blob signal score in this cycle, and adjust its
coordinate consistent as cycle 1.
"""


from numpy import (around, transpose, nonzero)
from scipy.stats import binom_test


def image_model_pooling_Ke(f_image_model_A, f_image_model_T, f_image_model_C, f_image_model_G):
    """
    :param f_image_model_A: Channel A in a cycle from the 3D common data tensor.
    :param f_image_model_T: Channel T in a cycle from the 3D common data tensor.
    :param f_image_model_C: Channel C in a cycle from the 3D common data tensor.
    :param f_image_model_G: Channel G in a cycle from the 3D common data tensor.
    :return f_image_model_pool: A dictionary of blobs, all the types of base in a certain location, including the
    coordinates, bases and their base scores.
    """
    f_image_model_pool = {}

    #################################################
    # Collect all the coordinates of detected blobs #
    #################################################
    f_image_model = f_image_model_A + f_image_model_T + f_image_model_C + f_image_model_G
    #################################################

    ##############################################################################################################
    # Each coordinate stores the base scores, and the largest one is made to as the representative of this cycle #
    # Other channels, which take non-largest base scores, will be following used to calculate the Quality of     #
    # their coordinates                                                                                          #
    ##############################################################################################################
    for row, col in transpose(nonzero(f_image_model)):
        #######################################################################################################
        # Our software just handle the image with its size smaller than 99999x99999                           #
        # This size limit should fit most of images                                                           #
        # You can modify this limit like following options in each place of 'read_id' for fitting your images #
        #######################################################################################################
        read_id = 'r%05dc%05d' % (row + 1, col + 1)
        ########
        # read_id = 'r%06dc%06d' % (row + 1, col + 1)  # Alternative option
        ########
        # read_id = 'r%07dc%07d' % (row + 1, col + 1)  # Alternative option
        ########
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
    :param f_image_model_1: Channel 1 in a cycle from the 3D common data tensor.
    :param f_image_model_2: Channel 2 in a cycle from the 3D common data tensor.
    :param f_image_model_3: Channel 3 in a cycle from the 3D common data tensor.
    :param f_image_model_4: Channel 4 in a cycle from the 3D common data tensor.
    :param f_image_model_5: Channel 5 in a cycle from the 3D common data tensor.
    :param f_image_model_6: Channel 6 in a cycle from the 3D common data tensor.
    :param f_image_model_7: Channel 7 in a cycle from the 3D common data tensor.
    :param f_image_model_8: Channel 8 in a cycle from the 3D common data tensor.
    :param f_image_model_9: Channel 9 in a cycle from the 3D common data tensor.
    :param f_image_model_A: Channel A in a cycle from the 3D common data tensor.
    :param f_image_model_B: Channel B in a cycle from the 3D common data tensor.
    :param f_image_model_C: Channel C in a cycle from the 3D common data tensor.
    :return f_image_model_pool: A dictionary of blobs, all the types of base in a certain location, including the
    coordinate, bases and their base scores.
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


def pool2base(f_image_model_pool):
    """
    :param f_image_model_pool: The dictionary of blobs, including the base and the coordinates.
    :return f_base_box: The dictionary of bases, only one representative in a certain location, including the
    coordinate, base and its error rates.
    """
    f_base_box = {}

    for read_id in f_image_model_pool:
        sorted_base = [_ for _ in sorted(f_image_model_pool[read_id].items(), key=lambda x: x[1], reverse=True)]

        if sorted_base[0][1] > sorted_base[1][1]:
            error_rate = around(binom_test((sorted_base[0][1], sorted_base[1][1]), p=0.5, alternative='greater'), 4)

            if read_id not in f_base_box:
                f_base_box.update({read_id: [sorted_base[0][0], error_rate]})

    return f_base_box


if __name__ == '__main__':
    pass
