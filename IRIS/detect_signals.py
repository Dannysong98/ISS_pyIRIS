#!/usr/bin/env python3
"""
This class is used to detect fluorescence signal in each channel.

Usually, chemical reaction or taking photo in local region will trigger the generating of different quality
fluorescence signal in a image, such as low gray scale or indistinction between fluorescence signal and background.

A Morphological transformation is called to expose a majority of high quality fluorescence signal as blobs. In this
process, large scale transformator is used to expose dissociative fluorescence signal and small one is used to treat
the adjacent signal as accurately as possible.

When fluorescence signals are exposed, a simple blob detection algorithm is called for blob locating. In our
practice, not only dense fluorescence signal but also sparse blob can be detected by this parameter-optimized
algorithm, while the ambiguous ones will be abandoned. After detection, for each detected blobs, blob's base
score, which is calculated by their gray scale in core (4x4) region being subtracted by surrounding (10x10), is
recorded to calculate base quality in the next step.
"""


from cv2 import (getStructuringElement, morphologyEx, GaussianBlur, convertScaleAbs,
                 SimpleBlobDetector, SimpleBlobDetector_Params,
                 MORPH_ELLIPSE, MORPH_TOPHAT)
######################
# Alternative option #
######################
# from cv2 import (Laplacian,
#                  CV_32F)
######################
from numpy import (array, zeros, ones, sum, divide, around, abs, max, int, float32, uint8, bool_, fft)
from scipy.stats import mode

from .call_bases import (image_model_pooling_Ke, image_model_pooling_Eng, image_model_pooling_Chen, pool2base)


def __hpf(f_img):
    """
    High-pass Filter

    :param f_img: Input image
    :return: Filtered image
    """
    row, col = f_img.shape

    masker_window = ones((row, col), dtype=bool_)
    masker_window[int(row / 2) - int(row * 0.2):int(row / 2) + int(row * 0.2),
                  int(col / 2) - int(col * 0.2):int(col / 2) + int(col * 0.2)] = 0

    f_img = abs(fft.ifft2(fft.ifftshift(fft.fftshift(fft.fft2(f_img)) * masker_window)))
    f_img = convertScaleAbs(f_img / max(f_img) * 255)

    return f_img


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

    greyscale_model_A = zeros(channel_A.shape, dtype=float32)
    greyscale_model_T = zeros(channel_T.shape, dtype=float32)
    greyscale_model_C = zeros(channel_C.shape, dtype=float32)
    greyscale_model_G = zeros(channel_G.shape, dtype=float32)

    ###############################################################################
    # Here, a morphological transformation, Tophat, under a 15x15 ELLIPSE kernel, #
    # is used to expose blobs                                                     #
    ###############################################################################
    ksize = (15, 15)
    kernel = getStructuringElement(MORPH_ELLIPSE, ksize)
    channel_A = morphologyEx(channel_A, MORPH_TOPHAT, kernel, iterations=3)
    channel_T = morphologyEx(channel_T, MORPH_TOPHAT, kernel, iterations=3)
    channel_C = morphologyEx(channel_C, MORPH_TOPHAT, kernel, iterations=3)
    channel_G = morphologyEx(channel_G, MORPH_TOPHAT, kernel, iterations=3)
    ########

    ###############################
    # Block of alternative option #
    ###############################
    # channel_A = convertScaleAbs(Laplacian(GaussianBlur(channel_A, (3, 3), 0), CV_32F))
    # channel_T = convertScaleAbs(Laplacian(GaussianBlur(channel_T, (3, 3), 0), CV_32F))
    # channel_C = convertScaleAbs(Laplacian(GaussianBlur(channel_C, (3, 3), 0), CV_32F))
    # channel_G = convertScaleAbs(Laplacian(GaussianBlur(channel_G, (3, 3), 0), CV_32F))
    ###############################

    ###############################################################################

    channel_list = (channel_A, channel_T, channel_C, channel_G)

    mor_kps1 = []
    mor_kps2 = []

    ##########################################################
    # Parameters setup for preliminary blob detection        #
    # Here, some of parameters are very crucial, such as     #
    # 'thresholdStep', 'minRepeatability', 'minArea', which  #
    # could greatly affect the number of detected blobs. And #
    # more importantly, they would need to be optimized in   #
    # different experiments.                                 #
    #                                                        #
    # We prepared some parameter set for the different       #
    # experiments during debugging, and we look forward to   #
    # standardize the experiments                            #
    ##########################################################
    blob_params = SimpleBlobDetector_Params()

    blob_params.minThreshold = 5

    blob_params.thresholdStep = 2
    blob_params.minRepeatability = 2
    ########
    # blob_params.thresholdStep = 3  # Alternative option
    # blob_params.minRepeatability = 3  # Alternative option

    blob_params.minDistBetweenBlobs = 2

    ####################################################################################
    # This parameter is used for filtering those extremely large blobs, which likely   #
    # to results from contamination                                                    #
    #                                                                                  #
    # Unfortunately, some genes expressing highly at a dense region tend to form large #
    # blobs thus would to be filtered, lead to optics-identification failure.          #
    # A known case in our practise is the gene 'pro-corazonin-like', this is a highly  #
    # expressed gene at a small region in brain of some insects, and is usually        #
    # detected as a low- or non-expression gene by IRIS                                #
    ####################################################################################
    blob_params.filterByArea = True

    blob_params.minArea = 1
    ########
    # blob_params.minArea = 4  # Alternative option

    blob_params.maxArea = 121
    ########
    # blob_params.maxArea = 65  # Alternative option
    # blob_params.maxArea = 145  # Alternative option
    ####################################################################################

    blob_params.filterByCircularity = True
    blob_params.minCircularity = 0.3

    blob_params.filterByConvexity = True
    blob_params.minConvexity = 0.1

    blob_params.filterByColor = True
    ##########################################################

    for img in channel_list:
        blob_params.blobColor = 0
        mor_detector = SimpleBlobDetector.create(blob_params)
        mor_kps1.extend(mor_detector.detect(255 - img))

        blob_params.blobColor = 255
        mor_detector = SimpleBlobDetector.create(blob_params)
        mor_kps2.extend(mor_detector.detect(img))

    mor_kps = mor_kps1 + mor_kps2

    #################################################################################
    # To map all the detected blobs into a new mask layer for redundancy filtering, #
    # and detect on this mask layer again to ensure blobs' location across all      #
    # channels in this cycle                                                        #
    #################################################################################
    mask_layer = zeros(channel_A.shape, dtype=uint8)

    for key_point in mor_kps:
        r = int(key_point.pt[1])
        c = int(key_point.pt[0])

        mask_layer[r:(r + 2), c:(c + 2)] = 255

    #############################################################################################################
    # If a Gaussian blur employed in here, these large blobs could be detected well, but high density blobs not #
    #############################################################################################################
    mask_layer = GaussianBlur(mask_layer, (3, 3), 0)
    #############################################################################################################

    detector = SimpleBlobDetector.create(blob_params)
    kps = detector.detect(mask_layer)
    #################################################################################

    ##########################################################################
    # Calculate the threshold for distinction between blobs and potential    #
    # pseudo-blobs                                                           #
    #                                                                        #
    # A crucial feature of real blob is that the gray-scale of pixel         #
    # should increase rapidly in its core region, compared with periphery    #
    #                                                                        #
    # The step of detection could expose a massive amount of blobs but also  #
    # include some false-positive. We calculate the difference of mean       #
    # gray-scale between pixel in core region and periphery of each blob,    #
    # which named as 'base score', and automatically determine the threshold #
    # for each channel. This threshold could be used to filter false-positive#
    # blobs in following step                                                #
    ##########################################################################
    diff_list_A = []
    diff_list_T = []
    diff_list_C = []
    diff_list_G = []

    for key_point in kps:
        r = int(key_point.pt[1])
        c = int(key_point.pt[0])

        diff_A = sum(channel_A[(r - 1):(r + 3), (c - 1):(c + 3)]) / 16 - \
                 sum(channel_A[(r - 4):(r + 6), (c - 4):(c + 6)]) / 100

        diff_T = sum(channel_T[(r - 1):(r + 3), (c - 1):(c + 3)]) / 16 - \
                 sum(channel_T[(r - 4):(r + 6), (c - 4):(c + 6)]) / 100

        diff_C = sum(channel_C[(r - 1):(r + 3), (c - 1):(c + 3)]) / 16 - \
                 sum(channel_C[(r - 4):(r + 6), (c - 4):(c + 6)]) / 100

        diff_G = sum(channel_G[(r - 1):(r + 3), (c - 1):(c + 3)]) / 16 - \
                 sum(channel_G[(r - 4):(r + 6), (c - 4):(c + 6)]) / 100

        if diff_A > 0:
            diff_list_A.append(int(around(diff_A)))

        if diff_T > 0:
            diff_list_T.append(int(around(diff_T)))

        if diff_C > 0:
            diff_list_C.append(int(around(diff_C)))

        if diff_G > 0:
            diff_list_G.append(int(around(diff_G)))

    diff_bk = 10

    cut_off_A = int(mode(around(divide(array(diff_list_A, dtype=uint8), diff_bk)) * diff_bk)[0][0]) - diff_bk / 2
    cut_off_T = int(mode(around(divide(array(diff_list_T, dtype=uint8), diff_bk)) * diff_bk)[0][0]) - diff_bk / 2
    cut_off_C = int(mode(around(divide(array(diff_list_C, dtype=uint8), diff_bk)) * diff_bk)[0][0]) - diff_bk / 2
    cut_off_G = int(mode(around(divide(array(diff_list_G, dtype=uint8), diff_bk)) * diff_bk)[0][0]) - diff_bk / 2
    print(cut_off_A, cut_off_T, cut_off_C, cut_off_G)
    #########################################################################

    ###################################################################################################
    # The coordinates of real blobs will be used to calculate the base score among different channels #
    ###################################################################################################
    for key_point in kps:
        r = int(key_point.pt[1])
        c = int(key_point.pt[0])

        if sum(channel_A[(r - 1):(r + 3), (c - 1):(c + 3)]) / 16 - \
                sum(channel_A[(r - 4):(r + 6), (c - 4):(c + 6)]) / 100 > cut_off_A:
            greyscale_model_A[r, c] = sum(channel_A[(r - 1):(r + 3), (c - 1):(c + 3)]) / 16 - \
                                      sum(channel_A[(r - 4):(r + 6), (c - 4):(c + 6)]) / 100

        if sum(channel_T[(r - 1):(r + 3), (c - 1):(c + 3)]) / 16 - \
                sum(channel_T[(r - 4):(r + 6), (c - 4):(c + 6)]) / 100 > cut_off_T:
            greyscale_model_T[r, c] = sum(channel_T[(r - 1):(r + 3), (c - 1):(c + 3)]) / 16 - \
                                      sum(channel_T[(r - 4):(r + 6), (c - 4):(c + 6)]) / 100

        if sum(channel_C[(r - 1):(r + 3), (c - 1):(c + 3)]) / 16 - \
                sum(channel_C[(r - 4):(r + 6), (c - 4):(c + 6)]) / 100 > cut_off_C:
            greyscale_model_C[r, c] = sum(channel_C[(r - 1):(r + 3), (c - 1):(c + 3)]) / 16 - \
                                      sum(channel_C[(r - 4):(r + 6), (c - 4):(c + 6)]) / 100

        if sum(channel_G[(r - 1):(r + 3), (c - 1):(c + 3)]) / 16 - \
                sum(channel_G[(r - 4):(r + 6), (c - 4):(c + 6)]) / 100 > cut_off_G:
            greyscale_model_G[r, c] = sum(channel_G[(r - 1):(r + 3), (c - 1):(c + 3)]) / 16 - \
                                      sum(channel_G[(r - 4):(r + 6), (c - 4):(c + 6)]) / 100
    ##################################################################################################

    image_model_pool = image_model_pooling_Ke(greyscale_model_A,
                                              greyscale_model_T,
                                              greyscale_model_C,
                                              greyscale_model_G)

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
    #########################################################################
    # The barcode should be encode as 1~9 and A, B, C                       #
    # N means ambiguous base                                                #
    # This strategy of encoding can make the data structure compatible with #
    # Ke's data                                                             #
    #########################################################################
    channel_1 = f_cycle[0]
    channel_2 = f_cycle[1]
    channel_3 = f_cycle[2]
    channel_4 = f_cycle[3]
    channel_5 = f_cycle[4]
    channel_6 = f_cycle[5]
    channel_7 = f_cycle[6]
    channel_8 = f_cycle[7]
    channel_9 = f_cycle[8]
    channel_A = f_cycle[9]
    channel_B = f_cycle[10]
    channel_C = f_cycle[11]
    #########################################################################

    greyscale_model_1 = zeros(channel_1.shape, dtype=float32)
    greyscale_model_2 = zeros(channel_2.shape, dtype=float32)
    greyscale_model_3 = zeros(channel_3.shape, dtype=float32)
    greyscale_model_4 = zeros(channel_4.shape, dtype=float32)
    greyscale_model_5 = zeros(channel_5.shape, dtype=float32)
    greyscale_model_6 = zeros(channel_6.shape, dtype=float32)
    greyscale_model_7 = zeros(channel_7.shape, dtype=float32)
    greyscale_model_8 = zeros(channel_8.shape, dtype=float32)
    greyscale_model_9 = zeros(channel_9.shape, dtype=float32)
    greyscale_model_A = zeros(channel_A.shape, dtype=float32)
    greyscale_model_B = zeros(channel_B.shape, dtype=float32)
    greyscale_model_C = zeros(channel_C.shape, dtype=float32)

    # ksize = (3, 3)
    # kernel = getStructuringElement(MORPH_ELLIPSE, ksize)
    #
    # channel_1 = morphologyEx(channel_1, MORPH_TOPHAT, kernel)
    # channel_2 = morphologyEx(channel_2, MORPH_TOPHAT, kernel)
    # channel_3 = morphologyEx(channel_3, MORPH_TOPHAT, kernel)
    # channel_4 = morphologyEx(channel_4, MORPH_TOPHAT, kernel)
    # channel_5 = morphologyEx(channel_5, MORPH_TOPHAT, kernel)
    # channel_6 = morphologyEx(channel_6, MORPH_TOPHAT, kernel)
    # channel_7 = morphologyEx(channel_7, MORPH_TOPHAT, kernel)
    # channel_8 = morphologyEx(channel_8, MORPH_TOPHAT, kernel)
    # channel_9 = morphologyEx(channel_9, MORPH_TOPHAT, kernel)
    # channel_A = morphologyEx(channel_A, MORPH_TOPHAT, kernel)
    # channel_B = morphologyEx(channel_B, MORPH_TOPHAT, kernel)
    # channel_C = morphologyEx(channel_C, MORPH_TOPHAT, kernel)
    ########

    ###############################
    # Block of alternative option #
    ###############################
    # channel_1 = convertScaleAbs(Laplacian(GaussianBlur(channel_1, (3, 3), 0), CV_32F))
    # channel_2 = convertScaleAbs(Laplacian(GaussianBlur(channel_2, (3, 3), 0), CV_32F))
    # channel_3 = convertScaleAbs(Laplacian(GaussianBlur(channel_3, (3, 3), 0), CV_32F))
    # channel_4 = convertScaleAbs(Laplacian(GaussianBlur(channel_4, (3, 3), 0), CV_32F))
    # channel_5 = convertScaleAbs(Laplacian(GaussianBlur(channel_5, (3, 3), 0), CV_32F))
    # channel_6 = convertScaleAbs(Laplacian(GaussianBlur(channel_6, (3, 3), 0), CV_32F))
    # channel_7 = convertScaleAbs(Laplacian(GaussianBlur(channel_7, (3, 3), 0), CV_32F))
    # channel_8 = convertScaleAbs(Laplacian(GaussianBlur(channel_8, (3, 3), 0), CV_32F))
    # channel_9 = convertScaleAbs(Laplacian(GaussianBlur(channel_9, (3, 3), 0), CV_32F))
    # channel_A = convertScaleAbs(Laplacian(GaussianBlur(channel_A, (3, 3), 0), CV_32F))
    # channel_B = convertScaleAbs(Laplacian(GaussianBlur(channel_B, (3, 3), 0), CV_32F))
    # channel_C = convertScaleAbs(Laplacian(GaussianBlur(channel_C, (3, 3), 0), CV_32F))
    ###############################

    channel_list = (channel_1, channel_2, channel_3,
                    channel_4, channel_5, channel_6,
                    channel_7, channel_8, channel_9,
                    channel_A, channel_B, channel_C)

    mor_kps1 = []
    mor_kps2 = []

    blob_params = SimpleBlobDetector_Params()

    blob_params.minThreshold = 5

    blob_params.thresholdStep = 2
    blob_params.minRepeatability = 2
    ########
    # blob_params.thresholdStep = 3  # Alternative option
    # blob_params.minRepeatability = 3  # Alternative option

    blob_params.minDistBetweenBlobs = 1

    blob_params.filterByArea = True

    blob_params.minArea = 1
    ########
    # blob_params.minArea = 2  # Alternative option

    blob_params.maxArea = 10
    ########
    # blob_params.maxArea = 100  # Alternative option

    blob_params.filterByCircularity = True
    blob_params.minCircularity = 0.3

    blob_params.filterByConvexity = True
    blob_params.minConvexity = 0.1

    blob_params.filterByColor = True

    for img in channel_list:
        blob_params.blobColor = 0
        mor_detector = SimpleBlobDetector.create(blob_params)
        mor_kps1.extend(mor_detector.detect(255 - img))

        blob_params.blobColor = 255
        mor_detector = SimpleBlobDetector.create(blob_params)
        mor_kps2.extend(mor_detector.detect(img))

    mor_kps = mor_kps1 + mor_kps2

    mask_layer = zeros(channel_1.shape, dtype=uint8)

    for key_point in mor_kps:
        r = int(key_point.pt[1])
        c = int(key_point.pt[0])

        mask_layer[r:(r + 2), c:(c + 2)] = 255

    mask_layer = GaussianBlur(mask_layer, (3, 3), 0)

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

        diff_1 = sum(channel_1[r:(r + 2), c:(c + 2)]) / 4 - sum(channel_1[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36
        diff_2 = sum(channel_2[r:(r + 2), c:(c + 2)]) / 4 - sum(channel_2[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36
        diff_3 = sum(channel_3[r:(r + 2), c:(c + 2)]) / 4 - sum(channel_3[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36
        diff_4 = sum(channel_4[r:(r + 2), c:(c + 2)]) / 4 - sum(channel_4[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36
        diff_5 = sum(channel_5[r:(r + 2), c:(c + 2)]) / 4 - sum(channel_5[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36
        diff_6 = sum(channel_6[r:(r + 2), c:(c + 2)]) / 4 - sum(channel_6[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36
        diff_7 = sum(channel_7[r:(r + 2), c:(c + 2)]) / 4 - sum(channel_7[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36
        diff_8 = sum(channel_8[r:(r + 2), c:(c + 2)]) / 4 - sum(channel_8[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36
        diff_9 = sum(channel_9[r:(r + 2), c:(c + 2)]) / 4 - sum(channel_9[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36
        diff_A = sum(channel_A[r:(r + 2), c:(c + 2)]) / 4 - sum(channel_A[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36
        diff_B = sum(channel_B[r:(r + 2), c:(c + 2)]) / 4 - sum(channel_B[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36
        diff_C = sum(channel_C[r:(r + 2), c:(c + 2)]) / 4 - sum(channel_C[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36

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

    diff_bk = 10

    cut_off_1 = int(mode(around(divide(array(diff_list_1, dtype=uint8), diff_bk)) * diff_bk)[0][0]) - diff_bk
    cut_off_2 = int(mode(around(divide(array(diff_list_2, dtype=uint8), diff_bk)) * diff_bk)[0][0]) - diff_bk
    cut_off_3 = int(mode(around(divide(array(diff_list_3, dtype=uint8), diff_bk)) * diff_bk)[0][0]) - diff_bk
    cut_off_4 = int(mode(around(divide(array(diff_list_4, dtype=uint8), diff_bk)) * diff_bk)[0][0]) - diff_bk
    cut_off_5 = int(mode(around(divide(array(diff_list_5, dtype=uint8), diff_bk)) * diff_bk)[0][0]) - diff_bk
    cut_off_6 = int(mode(around(divide(array(diff_list_6, dtype=uint8), diff_bk)) * diff_bk)[0][0]) - diff_bk
    cut_off_7 = int(mode(around(divide(array(diff_list_7, dtype=uint8), diff_bk)) * diff_bk)[0][0]) - diff_bk
    cut_off_8 = int(mode(around(divide(array(diff_list_8, dtype=uint8), diff_bk)) * diff_bk)[0][0]) - diff_bk
    cut_off_9 = int(mode(around(divide(array(diff_list_9, dtype=uint8), diff_bk)) * diff_bk)[0][0]) - diff_bk
    cut_off_A = int(mode(around(divide(array(diff_list_A, dtype=uint8), diff_bk)) * diff_bk)[0][0]) - diff_bk
    cut_off_B = int(mode(around(divide(array(diff_list_B, dtype=uint8), diff_bk)) * diff_bk)[0][0]) - diff_bk
    cut_off_C = int(mode(around(divide(array(diff_list_C, dtype=uint8), diff_bk)) * diff_bk)[0][0]) - diff_bk

    for key_point in kps:
        r = int(key_point.pt[1])
        c = int(key_point.pt[0])

        if sum(channel_1[r:(r + 2), c:(c + 2)]) / 4 - sum(channel_1[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 > cut_off_1:
            greyscale_model_1[r, c] = sum(channel_1[r:(r + 2), c:(c + 2)]) / 4 - \
                                      sum(channel_1[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36

        if sum(channel_2[r:(r + 2), c:(c + 2)]) / 4 - sum(channel_2[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 > cut_off_2:
            greyscale_model_2[r, c] = sum(channel_2[r:(r + 2), c:(c + 2)]) / 4 - \
                                      sum(channel_2[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36

        if sum(channel_3[r:(r + 2), c:(c + 2)]) / 4 - sum(channel_3[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 > cut_off_3:
            greyscale_model_3[r, c] = sum(channel_3[r:(r + 2), c:(c + 2)]) / 4 - \
                                      sum(channel_3[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36

        if sum(channel_4[r:(r + 2), c:(c + 2)]) / 4 - sum(channel_4[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 > cut_off_4:
            greyscale_model_4[r, c] = sum(channel_4[r:(r + 2), c:(c + 2)]) / 4 - \
                                      sum(channel_4[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36

        if sum(channel_5[r:(r + 2), c:(c + 2)]) / 4 - sum(channel_5[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 > cut_off_5:
            greyscale_model_5[r, c] = sum(channel_5[r:(r + 2), c:(c + 2)]) / 4 - \
                                      sum(channel_5[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36

        if sum(channel_6[r:(r + 2), c:(c + 2)]) / 4 - sum(channel_6[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 > cut_off_6:
            greyscale_model_6[r, c] = sum(channel_6[r:(r + 2), c:(c + 2)]) / 4 - \
                                      sum(channel_6[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36

        if sum(channel_7[r:(r + 2), c:(c + 2)]) / 4 - sum(channel_7[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 > cut_off_7:
            greyscale_model_7[r, c] = sum(channel_7[r:(r + 2), c:(c + 2)]) / 4 - \
                                      sum(channel_7[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36

        if sum(channel_8[r:(r + 2), c:(c + 2)]) / 4 - sum(channel_8[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 > cut_off_8:
            greyscale_model_8[r, c] = sum(channel_8[r:(r + 2), c:(c + 2)]) / 4 - \
                                      sum(channel_8[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36

        if sum(channel_9[r:(r + 2), c:(c + 2)]) / 4 - sum(channel_9[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 > cut_off_9:
            greyscale_model_9[r, c] = sum(channel_9[r:(r + 2), c:(c + 2)]) / 4 - \
                                      sum(channel_9[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36

        if sum(channel_A[r:(r + 2), c:(c + 2)]) / 4 - sum(channel_A[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 > cut_off_A:
            greyscale_model_A[r, c] = sum(channel_A[r:(r + 2), c:(c + 2)]) / 4 - \
                                      sum(channel_A[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36

        if sum(channel_B[r:(r + 2), c:(c + 2)]) / 4 - sum(channel_B[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 > cut_off_B:
            greyscale_model_B[r, c] = sum(channel_B[r:(r + 2), c:(c + 2)]) / 4 - \
                                      sum(channel_B[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36

        if sum(channel_C[r:(r + 2), c:(c + 2)]) / 4 - sum(channel_C[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 > cut_off_C:
            greyscale_model_C[r, c] = sum(channel_C[r:(r + 2), c:(c + 2)]) / 4 - \
                                      sum(channel_C[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36

    image_model_pool = image_model_pooling_Eng(greyscale_model_1, greyscale_model_2, greyscale_model_3,
                                               greyscale_model_4, greyscale_model_5, greyscale_model_6,
                                               greyscale_model_7, greyscale_model_8, greyscale_model_9,
                                               greyscale_model_A, greyscale_model_B, greyscale_model_C)

    base_box_in_one_cycle = pool2base(image_model_pool)

    return base_box_in_one_cycle


def detect_blobs_Lee(f_cycle):
    """
    For detect the fluorescence signal.

    Input registered image from different channels.
    Returning the grey scale model.

    :param f_cycle: A image matrix in the 3D common data tensor.
    :return: A base box of this cycle, which store their coordinates, base and its error rate.
    """
    channel_0 = f_cycle[0]
    channel_A = f_cycle[1]
    channel_T = f_cycle[2]
    channel_C = f_cycle[3]
    channel_G = f_cycle[4]

    greyscale_model_A = zeros(channel_A.shape, dtype=float32)
    greyscale_model_T = zeros(channel_T.shape, dtype=float32)
    greyscale_model_C = zeros(channel_C.shape, dtype=float32)
    greyscale_model_G = zeros(channel_G.shape, dtype=float32)

    #############################################################################
    # Here, a morphological transformation, Tophat, under a 15x15 ELLIPSE kernel, #
    # is used to expose blobs                                                   #
    #############################################################################
    # ksize = (15, 15)
    # kernel = getStructuringElement(MORPH_ELLIPSE, ksize)
    # channel_0 = morphologyEx(channel_0, MORPH_TOPHAT, kernel, iterations=3)
    ########

    ###############################
    # Block of alternative option #
    ###############################
    # channel_0 = convertScaleAbs(Laplacian(GaussianBlur(channel_0, (3, 3), 0), CV_32F))
    ###############################

    #############################################################################

    channel_list = (channel_0,)

    mor_kps1 = []
    mor_kps2 = []

    ##########################################################
    # Parameters setup for preliminary blob detection        #
    # Here, some of parameters are very crucial, such as     #
    # 'thresholdStep', 'minRepeatability', 'minArea', which  #
    # could greatly affect the number of detected blobs. And #
    # more importantly, they would need to be modified in    #
    # different experiments.                                 #
    #                                                        #
    # We prepared some cases for the different experiments   #
    # we met during debugging, and we look forward to        #
    # standardize the experiments                            #
    ##########################################################
    blob_params = SimpleBlobDetector_Params()

    blob_params.minThreshold = 5

    blob_params.thresholdStep = 2
    blob_params.minRepeatability = 2
    ########
    # blob_params.thresholdStep = 3  # Alternative option
    # blob_params.minRepeatability = 3  # Alternative option

    blob_params.minDistBetweenBlobs = 1

    ####################################################################################
    # This parameter is used for filtering those extremely large blobs, which likely   #
    # to results from contamination                                                    #
    #                                                                                  #
    # Unfortunately, some genes expressing highly at a dense region tend to form large #
    # blobs thus would to be filtered, lead to optics-identification failure.          #
    # A known case in our practise is the gene 'pro-corazonin-like', this is a highly  #
    # expressed gene at a small region in brain of some insects, and is usually        #
    # detected as a low- or non-expression gene in IRIS                                #
    ####################################################################################
    blob_params.filterByArea = True

    blob_params.minArea = 1
    ########
    # blob_params.minArea = 4  # Alternative option

    blob_params.maxArea = 10
    ########
    # blob_params.maxArea = 121  # Alternative option
    # blob_params.maxArea = 145  # Alternative option
    ####################################################################################

    blob_params.filterByCircularity = True
    blob_params.minCircularity = 0.3

    blob_params.filterByConvexity = True
    blob_params.minConvexity = 0.1

    blob_params.filterByColor = True
    ##########################################################

    for img in channel_list:
        blob_params.blobColor = 0
        mor_detector = SimpleBlobDetector.create(blob_params)
        mor_kps1.extend(mor_detector.detect(255 - img))

        blob_params.blobColor = 255
        mor_detector = SimpleBlobDetector.create(blob_params)
        mor_kps2.extend(mor_detector.detect(img))

    mor_kps = mor_kps1 + mor_kps2

    #################################################################################
    # To map all the detected blobs into a new mask layer for redundancy filtering, #
    # and detect on this mask layer again to ensure blobs' location across all      #
    # channels in this cycle                                                        #
    #################################################################################
    mask_layer = zeros(channel_0.shape, dtype=uint8)

    for key_point in mor_kps:
        r = int(key_point.pt[1])
        c = int(key_point.pt[0])

        mask_layer[r:(r + 2), c:(c + 2)] = 255

    mask_layer = GaussianBlur(mask_layer, (3, 3), 0)

    detector = SimpleBlobDetector.create(blob_params)
    kps = detector.detect(mask_layer)

    diff_list_A = []
    diff_list_T = []
    diff_list_C = []
    diff_list_G = []
    #################################################################################

    #########################################################################
    # Calculate the threshold for distinction between blobs and potential   #
    # pseudo-blobs                                                          #
    #                                                                       #
    # A crucial feature of real blob is that the gray-scale of pixel        #
    # should increase rapidly in its core region, compared with periphery   #
    #                                                                       #
    # The step of detection could expose a massive amount of blobs but also #
    # include some false-positive. We calculate the difference of mean      #
    # gray-scale between pixel in core region and periphery of each blob,   #
    # which named as 'base score', and calculate a threshold of each        #
    # channel. This threshold could be used to filter those false-positive  #
    # blobs in following step                                               #
    #########################################################################
    for key_point in kps:
        r = int(key_point.pt[1])
        c = int(key_point.pt[0])

        diff_A = sum(channel_A[r:(r + 2), c:(c + 2)]) / 4 - sum(channel_A[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36
        diff_T = sum(channel_T[r:(r + 2), c:(c + 2)]) / 4 - sum(channel_T[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36
        diff_C = sum(channel_C[r:(r + 2), c:(c + 2)]) / 4 - sum(channel_C[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36
        diff_G = sum(channel_G[r:(r + 2), c:(c + 2)]) / 4 - sum(channel_G[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36

        if diff_A > 0:
            diff_list_A.append(int(around(diff_A)))

        if diff_T > 0:
            diff_list_T.append(int(around(diff_T)))

        if diff_C > 0:
            diff_list_C.append(int(around(diff_C)))

        if diff_G > 0:
            diff_list_G.append(int(around(diff_G)))

    diff_bk = 10

    cut_off_A = int(mode(around(divide(array(diff_list_A, dtype=uint8), diff_bk)) * diff_bk)[0][0]) - diff_bk
    cut_off_T = int(mode(around(divide(array(diff_list_T, dtype=uint8), diff_bk)) * diff_bk)[0][0]) - diff_bk
    cut_off_C = int(mode(around(divide(array(diff_list_C, dtype=uint8), diff_bk)) * diff_bk)[0][0]) - diff_bk
    cut_off_G = int(mode(around(divide(array(diff_list_G, dtype=uint8), diff_bk)) * diff_bk)[0][0]) - diff_bk
    #########################################################################

    ##############################################################################################################
    # The coordinates of real blobs will be used to locate the difference of gary-scale among different channels #
    ##############################################################################################################
    for key_point in kps:
        r = int(key_point.pt[1])
        c = int(key_point.pt[0])

        if sum(channel_A[r:(r + 2), c:(c + 2)]) / 4 - sum(channel_A[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 > cut_off_A:
            greyscale_model_A[r, c] = sum(channel_A[r:(r + 2), c:(c + 2)]) / 4 - \
                                      sum(channel_A[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36

        if sum(channel_T[r:(r + 2), c:(c + 2)]) / 4 - sum(channel_T[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 > cut_off_T:
            greyscale_model_T[r, c] = sum(channel_T[r:(r + 2), c:(c + 2)]) / 4 - \
                                      sum(channel_T[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36

        if sum(channel_C[r:(r + 2), c:(c + 2)]) / 4 - sum(channel_C[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 > cut_off_C:
            greyscale_model_C[r, c] = sum(channel_C[r:(r + 2), c:(c + 2)]) / 4 - \
                                      sum(channel_C[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36

        if sum(channel_G[r:(r + 2), c:(c + 2)]) / 4 - sum(channel_G[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 > cut_off_G:
            greyscale_model_G[r, c] = sum(channel_G[r:(r + 2), c:(c + 2)]) / 4 - \
                                      sum(channel_G[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36
    ##############################################################################################################

    image_model_pool = image_model_pooling_Ke(greyscale_model_A,
                                              greyscale_model_T,
                                              greyscale_model_C,
                                              greyscale_model_G)

    base_box_in_one_cycle = pool2base(image_model_pool)

    return base_box_in_one_cycle


def detect_blobs_Chen(f_cycle):
    """
    For detect the fluorescence signal.

    Input registered image from different channels.
    Returning the grey scale model.

    :param f_cycle: A image matrix in the 3D common data tensor.
    :return: A base box of this cycle, which store their coordinates, base and its error rate.
    """
    channel_0 = f_cycle[0]

    greyscale_model_0 = zeros(channel_0.shape, dtype=float32)

    #############################################################################
    # Here, a morphological transformation, Tophat, under a 5x5 ELLIPSE kernel, #
    # is used to expose blobs                                                   #
    #############################################################################
    ksize = (3, 3)
    kernel = getStructuringElement(MORPH_ELLIPSE, ksize)
    channel_0 = morphologyEx(channel_0, MORPH_TOPHAT, kernel, iterations=3)
    ########

    ##################################################################
    # Block of alternative option                                    #
    # High-pass filter in frequency domain of Fourier transformation #
    ##################################################################
    # channel_0 = __hpf(channel_0)
    ##################################################################

    #############################################################################

    channel_list = (channel_0,)

    mor_kps1 = []
    mor_kps2 = []

    ##########################################################
    # Parameters setup for preliminary blob detection        #
    # Here, some of parameters are very crucial, such as     #
    # 'thresholdStep', 'minRepeatability', 'minArea', which  #
    # could greatly affect the number of detected blobs. And #
    # more importantly, they would need to be modified in    #
    # different experiments.                                 #
    #                                                        #
    # We prepared some cases for the different experiments   #
    # we met during debugging, and we look forward to        #
    # standardize the experiments                            #
    ##########################################################
    blob_params = SimpleBlobDetector_Params()

    blob_params.minThreshold = 5

    blob_params.thresholdStep = 4
    blob_params.minRepeatability = 2
    ########
    # blob_params.thresholdStep = 3  # Alternative option
    # blob_params.minRepeatability = 3  # Alternative option

    blob_params.minDistBetweenBlobs = 1

    ####################################################################################
    # This parameter is used for filtering those extremely large blobs, which likely   #
    # to results from contamination                                                    #
    ####################################################################################
    blob_params.filterByArea = True

    blob_params.minArea = 1
    ########
    # blob_params.minArea = 4  # Alternative option

    blob_params.maxArea = 10
    ########
    # blob_params.maxArea = 121  # Alternative option
    # blob_params.maxArea = 145  # Alternative option
    ####################################################################################

    blob_params.filterByCircularity = True
    blob_params.minCircularity = 0.3

    blob_params.filterByConvexity = True
    blob_params.minConvexity = 0.1

    blob_params.filterByColor = True
    ##########################################################

    for img in channel_list:
        blob_params.blobColor = 0
        mor_detector = SimpleBlobDetector.create(blob_params)
        mor_kps1.extend(mor_detector.detect(255 - img))

        blob_params.blobColor = 255
        mor_detector = SimpleBlobDetector.create(blob_params)
        mor_kps2.extend(mor_detector.detect(img))

    mor_kps = mor_kps1 + mor_kps2

    #################################################################################
    # To map all the detected blobs into a new mask layer for redundancy filtering, #
    # and detect on this mask layer again to ensure blobs' location across all      #
    # channels in this cycle                                                        #
    #################################################################################
    mask_layer = zeros(channel_0.shape, dtype=uint8)

    for key_point in mor_kps:
        r = int(key_point.pt[1])
        c = int(key_point.pt[0])

        mask_layer[r:(r + 2), c:(c + 2)] = 255

    mask_layer = GaussianBlur(mask_layer, (3, 3), 0)

    detector = SimpleBlobDetector.create(blob_params)
    kps = detector.detect(mask_layer)
    #################################################################################

    #########################################################################
    # Calculate the threshold for distinction between blobs and potential   #
    # pseudo-blobs                                                          #
    #                                                                       #
    # A crucial feature of real blob is that the gray-scale of pixel        #
    # should increase rapidly in its core region, compared with periphery   #
    #                                                                       #
    # The step of detection could expose a massive amount of blobs but also #
    # include some false-positive. We calculate the difference of mean      #
    # gray-scale between pixel in core region and periphery of each blob,   #
    # which named as 'base score', and calculate a threshold of each        #
    # channel. This threshold could be used to filter those false-positive  #
    # blobs in following step                                               #
    #########################################################################
    diff_list_0 = []

    for key_point in kps:
        r = int(key_point.pt[1])
        c = int(key_point.pt[0])

        diff_0 = sum(channel_0[r:(r + 2), c:(c + 2)]) / 4 - sum(channel_0[(r - 1):(r + 3), (c - 1):(c + 3)]) / 16

        if diff_0 > 0:
            diff_list_0.append(int(around(diff_0)))

    diff_bk = 10

    cut_off_0 = int(mode(around(divide(array(diff_list_0, dtype=uint8), diff_bk)))[0][0]) * diff_bk - diff_bk
    #########################################################################

    ##############################################################################################################
    # The coordinates of real blobs will be used to locate the difference of gary-scale among different channels #
    ##############################################################################################################
    for key_point in kps:
        r = int(key_point.pt[1])
        c = int(key_point.pt[0])

        if sum(channel_0[r:(r + 2), c:(c + 2)]) / 4 - sum(channel_0[(r - 1):(r + 3), (c - 1):(c + 3)]) / 16 > cut_off_0:
            greyscale_model_0[r, c] = sum(channel_0[r:(r + 2), c:(c + 2)]) / 4 - \
                                      sum(channel_0[(r - 1):(r + 3), (c - 1):(c + 3)]) / 16
    ##############################################################################################################

    image_model_pool = image_model_pooling_Chen(greyscale_model_0)

    base_box_in_one_cycle = pool2base(image_model_pool)

    return base_box_in_one_cycle


if __name__ == '__main__':
    pass
