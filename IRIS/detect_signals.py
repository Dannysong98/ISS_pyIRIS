#!/usr/bin/env python3
"""
This class is used to detect fluorescence signal in each signal channel.

Usually, fault of chemical reaction or taking photo in local region will trigger the generating of
different quality fluorescence signal in a image, like low of gray scale or indistinctiveness between
fluorescence signal and background.

Our workflow provide a double strategy to treat these kinds of situation above. Two scale Morphological (TopHat)
transformations are invoked to expose a majority of high quality fluorescence signal as blobs. In which, large
scale transformator is used to expose dissociative fluorescence signal and small one used to treat the adjacent
signal as accurately as possible.

When fluorescence signal exposed, a simple blob detection algorithm is invoked for blob locating. In our
practice, not only dense fluorescence signal but also sparse blob can be detected by this parameters optimized
algorithm, while the ambiguous one will be abandoned. After detection, for each detected blobs, the blob signal
score, which is calculated by their gray scale in core (3x3) region being divided by surrounding (9x9), is
recorded to be made as the measure of significance, it is also the base of called base quality in next step.
"""


from cv2.cv2 import (getStructuringElement, morphologyEx, GaussianBlur,
                     SimpleBlobDetector, SimpleBlobDetector_Params,
                     MORPH_ELLIPSE, MORPH_TOPHAT)
from numpy import (array, zeros, reshape,
                   sum, divide, floor, around,
                   float32, uint8)
from scipy.stats import mode

from ._call_bases import image_model_pooling_Ke, image_model_pooling_Eng, pool2base


def detect_blobs_Ke(f_cycle):
    """
    For detect the fluorescence signal.

    Input registered image from different channels.
    Returning the grey scale model.

    :param f_cycle: A image matrix in the 3D common data tensor.
    :return: A base box of this cycle, which store their coordinates, base and its error rate.
    """
    channel_A = f_cycle[0]
    channel_T = f_cycle[1]
    channel_C = f_cycle[2]
    channel_G = f_cycle[3]

    f_greyscale_model_A = zeros(channel_A.shape, dtype=float32)
    f_greyscale_model_T = zeros(channel_T.shape, dtype=float32)
    f_greyscale_model_C = zeros(channel_C.shape, dtype=float32)
    f_greyscale_model_G = zeros(channel_G.shape, dtype=float32)

    ksize = (15, 15)
    kernel = getStructuringElement(MORPH_ELLIPSE, ksize)

    channel_A = morphologyEx(channel_A, MORPH_TOPHAT, kernel, iterations=3)
    channel_T = morphologyEx(channel_T, MORPH_TOPHAT, kernel, iterations=3)
    channel_C = morphologyEx(channel_C, MORPH_TOPHAT, kernel, iterations=3)
    channel_G = morphologyEx(channel_G, MORPH_TOPHAT, kernel, iterations=3)

    channel_list = (channel_A, channel_T, channel_C, channel_G)

    mor_kps = []

    blob_params = SimpleBlobDetector_Params()

    blob_params.thresholdStep = 2
    blob_params.minRepeatability = 2
    blob_params.minDistBetweenBlobs = 2

    blob_params.filterByColor = True
    blob_params.blobColor = 255

    blob_params.filterByArea = True
    blob_params.minArea = 2
    blob_params.maxArea = 196

    blob_params.filterByCircularity = False
    blob_params.filterByConvexity = True

    for img in channel_list:
        blob_params.minThreshold = mode(floor(reshape(img, (img.size,)) / 2) * 2)[0][0]

        mor_detector = SimpleBlobDetector.create(blob_params)

        mor_kps.extend(mor_detector.detect(img))

    mask_layer = zeros(channel_A.shape, dtype=uint8)

    for key_point in mor_kps:
        r = int(key_point.pt[1])
        c = int(key_point.pt[0])

        mask_layer[r:(r + 2), c:(c + 2)] = 255

    mask_layer = GaussianBlur(mask_layer, (5, 5), 0)

    blob_params.minThreshold = 1

    detector = SimpleBlobDetector.create(blob_params)

    kps = detector.detect(mask_layer)

    diff_list_A = []
    diff_list_T = []
    diff_list_C = []
    diff_list_G = []

    for key_point in kps:
        r = int(key_point.pt[1])
        c = int(key_point.pt[0])

        diff_A = sum(channel_A[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                 sum(channel_A[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144

        diff_T = sum(channel_T[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                 sum(channel_T[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144

        diff_C = sum(channel_C[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                 sum(channel_C[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144

        diff_G = sum(channel_G[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                 sum(channel_G[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144

        if diff_A > 0:
            diff_list_A.append(int(around(diff_A)))

        if diff_T > 0:
            diff_list_T.append(int(around(diff_T)))

        if diff_C > 0:
            diff_list_C.append(int(around(diff_C)))

        if diff_G > 0:
            diff_list_G.append(int(around(diff_G)))

    diff_break = 10

    cut_off_A = int(mode(around(divide(array(diff_list_A, dtype=uint8), diff_break)))[0][0]) - diff_break
    cut_off_T = int(mode(around(divide(array(diff_list_T, dtype=uint8), diff_break)))[0][0]) - diff_break
    cut_off_C = int(mode(around(divide(array(diff_list_C, dtype=uint8), diff_break)))[0][0]) - diff_break
    cut_off_G = int(mode(around(divide(array(diff_list_G, dtype=uint8), diff_break)))[0][0]) - diff_break

    for key_point in kps:
        r = int(key_point.pt[1])
        c = int(key_point.pt[0])

        if sum(channel_A[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                sum(channel_A[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144 > cut_off_A:
            f_greyscale_model_A[r, c] = sum(channel_A[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                                        sum(channel_A[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144

        if sum(channel_T[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                sum(channel_T[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144 > cut_off_T:
            f_greyscale_model_T[r, c] = sum(channel_T[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                                        sum(channel_T[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144

        if sum(channel_C[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                sum(channel_C[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144 > cut_off_C:
            f_greyscale_model_C[r, c] = sum(channel_C[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                                        sum(channel_C[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144

        if sum(channel_G[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                sum(channel_G[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144 > cut_off_G:
            f_greyscale_model_G[r, c] = sum(channel_G[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                                        sum(channel_G[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144

    image_model_pool = image_model_pooling_Ke(f_greyscale_model_A,
                                              f_greyscale_model_T,
                                              f_greyscale_model_C,
                                              f_greyscale_model_G)

    base_box_in_one_cycle = pool2base(image_model_pool)

    return base_box_in_one_cycle


def detect_blobs_Eng(f_cycle):
    """
    For detect the fluorescence signal.

    Input registered image from different channels.
    Returning the grey scale model.

    :param f_cycle: A image matrix in the 3D common data tensor.
    :return: A base box of this cycle, which store their coordinates, base and its error rate.
    """
    channel_1 = f_cycle[0][0]
    channel_2 = f_cycle[0][1]
    channel_3 = f_cycle[0][2]
    channel_4 = f_cycle[1][0]
    channel_5 = f_cycle[1][1]
    channel_6 = f_cycle[1][2]
    channel_7 = f_cycle[2][0]
    channel_8 = f_cycle[2][1]
    channel_9 = f_cycle[2][2]
    channel_A = f_cycle[3][0]
    channel_B = f_cycle[3][1]
    channel_C = f_cycle[3][2]

    f_greyscale_model_1 = zeros(channel_1.shape, dtype=float32)
    f_greyscale_model_2 = zeros(channel_2.shape, dtype=float32)
    f_greyscale_model_3 = zeros(channel_3.shape, dtype=float32)
    f_greyscale_model_4 = zeros(channel_4.shape, dtype=float32)
    f_greyscale_model_5 = zeros(channel_5.shape, dtype=float32)
    f_greyscale_model_6 = zeros(channel_6.shape, dtype=float32)
    f_greyscale_model_7 = zeros(channel_7.shape, dtype=float32)
    f_greyscale_model_8 = zeros(channel_8.shape, dtype=float32)
    f_greyscale_model_9 = zeros(channel_9.shape, dtype=float32)
    f_greyscale_model_A = zeros(channel_A.shape, dtype=float32)
    f_greyscale_model_B = zeros(channel_B.shape, dtype=float32)
    f_greyscale_model_C = zeros(channel_C.shape, dtype=float32)

    ksize = (15, 15)
    kernel = getStructuringElement(MORPH_ELLIPSE, ksize)

    channel_1 = morphologyEx(channel_1, MORPH_TOPHAT, kernel)
    channel_2 = morphologyEx(channel_2, MORPH_TOPHAT, kernel)
    channel_3 = morphologyEx(channel_3, MORPH_TOPHAT, kernel)
    channel_4 = morphologyEx(channel_4, MORPH_TOPHAT, kernel)
    channel_5 = morphologyEx(channel_5, MORPH_TOPHAT, kernel)
    channel_6 = morphologyEx(channel_6, MORPH_TOPHAT, kernel)
    channel_7 = morphologyEx(channel_7, MORPH_TOPHAT, kernel)
    channel_8 = morphologyEx(channel_8, MORPH_TOPHAT, kernel)
    channel_9 = morphologyEx(channel_9, MORPH_TOPHAT, kernel)
    channel_A = morphologyEx(channel_A, MORPH_TOPHAT, kernel)
    channel_B = morphologyEx(channel_B, MORPH_TOPHAT, kernel)
    channel_C = morphologyEx(channel_C, MORPH_TOPHAT, kernel)

    channel_list = (channel_1, channel_2, channel_3,
                    channel_4, channel_5, channel_6,
                    channel_7, channel_8, channel_9,
                    channel_A, channel_B, channel_C)

    mor_kps = []

    blob_params = SimpleBlobDetector_Params()

    blob_params.thresholdStep = 2
    blob_params.minRepeatability = 2
    blob_params.minDistBetweenBlobs = 2

    blob_params.filterByColor = True
    blob_params.blobColor = 255

    blob_params.filterByArea = True
    blob_params.minArea = 2
    blob_params.maxArea = 196

    blob_params.filterByCircularity = False
    blob_params.filterByConvexity = True

    for img in channel_list:
        blob_params.minThreshold = mode(floor(reshape(img, (img.size,)) / 2) * 2)[0][0]

        mor_detector = SimpleBlobDetector.create(blob_params)

        mor_kps.extend(mor_detector.detect(img))

    mask_layer = zeros(channel_1.shape, dtype=uint8)

    for key_point in mor_kps:
        r = int(key_point.pt[1])
        c = int(key_point.pt[0])

        mask_layer[r:(r + 2), c:(c + 2)] = 255

    mask_layer = GaussianBlur(mask_layer, (5, 5), 0)

    blob_params.minThreshold = 1

    detector = SimpleBlobDetector.create(blob_params)

    kps = detector.detect(mask_layer)

    diff_list_1 = []
    diff_list_2 = []
    diff_list_3 = []
    diff_list_4 = []
    diff_list_5 = []
    diff_list_6 = []
    diff_list_7 = []
    diff_list_8 = []
    diff_list_9 = []
    diff_list_A = []
    diff_list_B = []
    diff_list_C = []

    for key_point in kps:
        r = int(key_point.pt[1])
        c = int(key_point.pt[0])

        diff_1 = sum(channel_1[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                 sum(channel_1[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144

        diff_2 = sum(channel_2[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                 sum(channel_2[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144

        diff_3 = sum(channel_3[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                 sum(channel_3[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144

        diff_4 = sum(channel_4[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                 sum(channel_4[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144

        diff_5 = sum(channel_5[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                 sum(channel_5[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144

        diff_6 = sum(channel_6[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                 sum(channel_6[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144

        diff_7 = sum(channel_7[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                 sum(channel_7[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144

        diff_8 = sum(channel_8[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                 sum(channel_8[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144

        diff_9 = sum(channel_9[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                 sum(channel_9[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144

        diff_A = sum(channel_A[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                 sum(channel_A[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144

        diff_B = sum(channel_B[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                 sum(channel_B[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144

        diff_C = sum(channel_C[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                 sum(channel_C[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144

        if diff_1 > 0:
            diff_list_1.append(int(around(diff_1)))

        if diff_2 > 0:
            diff_list_2.append(int(around(diff_2)))

        if diff_3 > 0:
            diff_list_3.append(int(around(diff_3)))

        if diff_4 > 0:
            diff_list_4.append(int(around(diff_4)))

        if diff_5 > 0:
            diff_list_5.append(int(around(diff_5)))

        if diff_6 > 0:
            diff_list_6.append(int(around(diff_6)))

        if diff_7 > 0:
            diff_list_7.append(int(around(diff_7)))

        if diff_8 > 0:
            diff_list_8.append(int(around(diff_8)))

        if diff_9 > 0:
            diff_list_9.append(int(around(diff_9)))

        if diff_A > 0:
            diff_list_A.append(int(around(diff_A)))

        if diff_B > 0:
            diff_list_B.append(int(around(diff_B)))

        if diff_C > 0:
            diff_list_C.append(int(around(diff_C)))

    diff_break = 10

    cut_off_1 = int(mode(around(divide(array(diff_list_1, dtype=uint8), diff_break)))[0][0]) - diff_break
    cut_off_2 = int(mode(around(divide(array(diff_list_2, dtype=uint8), diff_break)))[0][0]) - diff_break
    cut_off_3 = int(mode(around(divide(array(diff_list_3, dtype=uint8), diff_break)))[0][0]) - diff_break
    cut_off_4 = int(mode(around(divide(array(diff_list_4, dtype=uint8), diff_break)))[0][0]) - diff_break
    cut_off_5 = int(mode(around(divide(array(diff_list_5, dtype=uint8), diff_break)))[0][0]) - diff_break
    cut_off_6 = int(mode(around(divide(array(diff_list_6, dtype=uint8), diff_break)))[0][0]) - diff_break
    cut_off_7 = int(mode(around(divide(array(diff_list_7, dtype=uint8), diff_break)))[0][0]) - diff_break
    cut_off_8 = int(mode(around(divide(array(diff_list_8, dtype=uint8), diff_break)))[0][0]) - diff_break
    cut_off_9 = int(mode(around(divide(array(diff_list_9, dtype=uint8), diff_break)))[0][0]) - diff_break
    cut_off_A = int(mode(around(divide(array(diff_list_A, dtype=uint8), diff_break)))[0][0]) - diff_break
    cut_off_B = int(mode(around(divide(array(diff_list_B, dtype=uint8), diff_break)))[0][0]) - diff_break
    cut_off_C = int(mode(around(divide(array(diff_list_C, dtype=uint8), diff_break)))[0][0]) - diff_break

    for key_point in kps:
        r = int(key_point.pt[1])
        c = int(key_point.pt[0])

        if sum(channel_1[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                sum(channel_1[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144 > cut_off_1:
            f_greyscale_model_1[r, c] = sum(channel_1[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                                        sum(channel_1[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144

        if sum(channel_2[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                sum(channel_2[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144 > cut_off_2:
            f_greyscale_model_2[r, c] = sum(channel_2[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                                        sum(channel_2[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144

        if sum(channel_3[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                sum(channel_3[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144 > cut_off_3:
            f_greyscale_model_3[r, c] = sum(channel_3[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                                        sum(channel_3[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144

        if sum(channel_4[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                sum(channel_4[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144 > cut_off_4:
            f_greyscale_model_4[r, c] = sum(channel_4[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                                        sum(channel_4[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144

        if sum(channel_5[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                sum(channel_5[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144 > cut_off_5:
            f_greyscale_model_5[r, c] = sum(channel_5[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                                        sum(channel_5[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144

        if sum(channel_6[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                sum(channel_6[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144 > cut_off_6:
            f_greyscale_model_6[r, c] = sum(channel_6[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                                        sum(channel_6[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144

        if sum(channel_7[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                sum(channel_7[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144 > cut_off_7:
            f_greyscale_model_7[r, c] = sum(channel_7[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                                        sum(channel_7[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144

        if sum(channel_8[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                sum(channel_8[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144 > cut_off_8:
            f_greyscale_model_8[r, c] = sum(channel_8[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                                        sum(channel_8[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144

        if sum(channel_9[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                sum(channel_9[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144 > cut_off_9:
            f_greyscale_model_9[r, c] = sum(channel_9[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                                        sum(channel_9[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144

        if sum(channel_A[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                sum(channel_A[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144 > cut_off_A:
            f_greyscale_model_A[r, c] = sum(channel_A[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                                        sum(channel_A[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144

        if sum(channel_B[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                sum(channel_B[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144 > cut_off_B:
            f_greyscale_model_B[r, c] = sum(channel_B[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                                        sum(channel_B[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144

        if sum(channel_C[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                sum(channel_C[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144 > cut_off_C:
            f_greyscale_model_C[r, c] = sum(channel_C[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                                        sum(channel_C[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144

    image_model_pool = image_model_pooling_Eng(f_greyscale_model_1, f_greyscale_model_2, f_greyscale_model_3,
                                               f_greyscale_model_4, f_greyscale_model_5, f_greyscale_model_6,
                                               f_greyscale_model_7, f_greyscale_model_8, f_greyscale_model_9,
                                               f_greyscale_model_A, f_greyscale_model_B, f_greyscale_model_C)

    base_box_in_one_cycle = pool2base(image_model_pool)

    return base_box_in_one_cycle


if __name__ == '__main__':
    pass
