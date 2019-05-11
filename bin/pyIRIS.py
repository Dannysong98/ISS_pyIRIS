#!/usr/bin/env python3


"""
    Description:    This software is used to transform the fluorescent signal of in situ sequencing (ISS) into base
                    sequences (barcode sequences), as well as the quality of base and their coordinates.

    Author: Hao Yu (yuhao@genomics.cn)
            Yang Zhou (zhouyang@genomics.cn)

    ChangeLog:  2019-01-15  r001    First development version
                2019-03-18  r002    Make Simple Blob Detector as the default detector
                2019-03-27  r003    To modify the detecting strategy
                2019-04-08  r004    To modify the quality algorithm
                2019-04-09  r005    To adjust the strategy of function of blob exposing
                2019-04-10  r006    To add the code document to explain primary functions
                2019-04-16  r007    1) To modify the algorithm of registration for getting more true positive blobs
                                    2) To modify the algorithm for blob detecting for accuracy
                                    3) To modify the parameter set of blob detector for accuracy
                2019-04-24  r008    To modify the algorithm for saturate blob exposing
                2019-04-25  r009    To modify the strategy for blob richly exposing and detection
                2019-05-09  r010    To modify the algorithm for blob exposing
"""

import re
import sys

import cv2 as cv
import numpy as np
import scipy.stats as sst


##################
# Data Structure #
##################

####################
# Class of Profile #
####################

class Profile:
    """This class is used to parse the profile of parameters."""

    def __init__(self, f_profile_path):

        self.__profile_path = f_profile_path
        self.__par_box = {'channel_A': 'Y5.tif', 'channel_T': 'GFP.tif', 'channel_C': 'TXR.tif', 'channel_G': 'Y3.tif',
                          'cycle_mask': ['#1', '#2', '#3'],
                          'root_path': './'}

        self.par_pool = {'baseImg_path_A': [],
                         'baseImg_path_T': [],
                         'baseImg_path_C': [],
                         'baseImg_path_G': []}

    def check_profile(self):

        regx = re.compile(r'^(\w+)\s*=\s*(.+)$')

        with open(self.__profile_path, 'r') as FH:
            for ln in FH:

                par = regx.findall(ln)[0][0]
                val = regx.findall(ln)[0][1]

                if par in self.__par_box:
                    self.__par_box[par] = val

                    if par == 'cycle_mask':

                        self.__par_box['cycle_mask'] = []

                        for cy in val.split():
                            self.__par_box['cycle_mask'].append(cy)

                else:
                    print('"' + par + '" is an invalid parameter', file=sys.stderr)

    def parse_profile(self):

        for cy in self.__par_box['cycle_mask']:

            self.par_pool['baseImg_path_A'].append(self.__par_box['root_path'] + cy + self.__par_box['channel_A'])
            self.par_pool['baseImg_path_T'].append(self.__par_box['root_path'] + cy + self.__par_box['channel_T'])
            self.par_pool['baseImg_path_C'].append(self.__par_box['root_path'] + cy + self.__par_box['channel_C'])
            self.par_pool['baseImg_path_G'].append(self.__par_box['root_path'] + cy + self.__par_box['channel_G'])


###############################
# Class of Image Registration #
###############################

class ImageRegistration:
    """
        This class is used to register the image from different cycles.

        This class include two internal methods and a public method. The internal method '__get_key_points' is used to
        capture key points in an image. Here, we invoke the BRISK (Leutenegger et al. 2011) as our core algorithm,
        because more key points can be captured in our practice, compared with other familiar methods, such as ORB
        (Rublee et al. 2011), SIFT (Lowe 2004), SURF (Bay et al. 2006), and it is free. Another internal method,
        '__get_good_match', is used to calculate the best matched pairs among the captured key points by their
        description.

        There two request inputs when its instantiation. The first one is a base image matrix and another one is the
        image matrix which need to be registered.
    """

    def __init__(self, f_query_img_mat, f_train_img_mat):

        self.__query_mat = f_query_img_mat
        self.__train_mat = f_train_img_mat

        self.registered_image = np.array([], dtype=np.uint8)
        self.matrix = np.zeros((3, 2), dtype=np.float32)

    @staticmethod
    def __get_key_points_and_descriptors(f_gray_img):

        kernel = cv.getStructuringElement(cv.MORPH_CROSS, (15, 15))
        f_gray_img = cv.morphologyEx(f_gray_img, cv.MORPH_GRADIENT, kernel)
        f_gray_img = cv.morphologyEx(f_gray_img, cv.MORPH_CLOSE, kernel, iterations=2)

        detector = cv.BRISK.create()
        extractor = cv.BRISK.create()

        key_points = detector.detect(f_gray_img) + \
                     detector.detect(cv.add(f_gray_img, f_gray_img))

        _, descriptors = extractor.compute(f_gray_img, key_points)

        return key_points, descriptors

    @staticmethod
    def __get_good_matchs(f_descriptor1, f_descriptor2):

        matcher = cv.BFMatcher.create(normType=cv.NORM_HAMMING, crossCheck=True)

        matches = matcher.knnMatch(f_descriptor1, f_descriptor2, 1)

        good_matches = [best_match_pair[0] for best_match_pair in matches if len(best_match_pair) > 0]

        good_matches = sorted(good_matches, key=lambda x: x.distance)

        return good_matches

    def register_image(self):

        kp1, des1 = self.__get_key_points_and_descriptors(self.__query_mat)
        kp2, des2 = self.__get_key_points_and_descriptors(self.__train_mat)

        good_matches = self.__get_good_matchs(des1, des2)

        if len(good_matches) >= 3:

            pts_a = np.float32([kp1[_.queryIdx].pt for _ in good_matches]).reshape(-1, 1, 2)
            pts_b = np.float32([kp2[_.trainIdx].pt for _ in good_matches]).reshape(-1, 1, 2)

            _, mask = cv.findHomography(pts_a, pts_b, cv.RANSAC)

            good_matches = [good_matches[k] for k in range(0, mask.size) if mask[k][0] == 1]

            pts_a_filtered = np.float32([kp1[_.queryIdx].pt for _ in good_matches]).reshape(-1, 1, 2)
            pts_b_filtered = np.float32([kp2[_.trainIdx].pt for _ in good_matches]).reshape(-1, 1, 2)

            self.matrix = cv.estimateRigidTransform(pts_a_filtered, pts_b_filtered, fullAffine=False)

            self.registered_image = cv.warpAffine(self.__train_mat, self.matrix,
                                                  (self.__query_mat.shape[1], self.__query_mat.shape[0]),
                                                  flags=cv.INTER_LANCZOS4 + cv.WARP_INVERSE_MAP)

        else:
            print('REGISTERING ERROR', file=sys.stderr)


#############################
# Class of Feature Detector #
#############################

class Detector:
    """
        This class is used to detect fluorescence signal in each signal channel.

        Usually, fault of chemical reaction or taking photo in local region will trigger the generating of
        different quality fluorescence signal in a image, like low of gray scale or indistinctiveness between
        fluorescence signal and background.

        Our workflow provide a double strategy to treat these kinds of situation above. Two scale Morphological (TopHat)
        transformations are invoked to expose a majority of high quality fluorescence signal as blobs. In which, large
        scale transformator is used to expose dissociative fluorescence signal and small one used to treat the adjacent
        signal as accurately as possible. Besides, a Laplacian of Gaussaian (LoG) taking 5x5 size kernel transformator
        is invoked to process the low quality region, and expose signal blob as more as possible.

        When fluorescence signal exposed, a simple blob detection algorithm is invoked for blob locating. In our
        practice, not only dense fluorescence signal but also sparse blob can be detected by this parameters optimized
        algorithm, while the ambiguous one will be abandoned. After detection, for each detected blobs, the blob signal
        score, which is calculated by their gray scale in core (3x3) region being divided by surrounding (9x9), is
        recorded to be made as the meassure of significance, it is also the base of called base quality in next step.
    """

    channel_list = []

    def __init__(self, f_img_taking_bg, f_img_taking_bg_A, f_img_taking_bg_T, f_img_taking_bg_C, f_img_taking_bg_G):

        self.__img_mat = f_img_taking_bg

        self.__img_mat_A = f_img_taking_bg_A
        self.__img_mat_T = f_img_taking_bg_T
        self.__img_mat_C = f_img_taking_bg_C
        self.__img_mat_G = f_img_taking_bg_G

        self.blob_greyscale_model_A = np.zeros(self.__img_mat.shape, dtype=np.float32)
        self.blob_greyscale_model_T = np.zeros(self.__img_mat.shape, dtype=np.float32)
        self.blob_greyscale_model_C = np.zeros(self.__img_mat.shape, dtype=np.float32)
        self.blob_greyscale_model_G = np.zeros(self.__img_mat.shape, dtype=np.float32)

    def blob_richly_detect(self):

        mor_img = cv.morphologyEx(self.__img_mat, cv.MORPH_TOPHAT,
                                  cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15)), iterations=2)

        img = cv.GaussianBlur(cv.add(self.__img_mat, mor_img), (5, 5), 0)

        blob_params = cv.SimpleBlobDetector_Params()

        blob_params.minThreshold = 1
        blob_params.thresholdStep = 2
        blob_params.minRepeatability = 2
        blob_params.minDistBetweenBlobs = 2

        blob_params.filterByArea = True
        blob_params.minArea = 1

        detector = cv.SimpleBlobDetector.create(blob_params)

        mor_kps = detector.detect(cv.bitwise_not(img))

        diff_list_A = []
        diff_list_T = []
        diff_list_C = []
        diff_list_G = []

        for key_point in mor_kps:

            r = int(key_point.pt[1])
            c = int(key_point.pt[0])

            diff_A = np.sum(self.__img_mat_A[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                     np.sum(self.__img_mat_A[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144

            diff_T = np.sum(self.__img_mat_T[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                     np.sum(self.__img_mat_T[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144

            diff_C = np.sum(self.__img_mat_C[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                     np.sum(self.__img_mat_C[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144

            diff_G = np.sum(self.__img_mat_G[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                     np.sum(self.__img_mat_G[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144

            diff_list_A.append(diff_A) if diff_A >= 0 else 0
            diff_list_T.append(diff_T) if diff_T >= 0 else 0
            diff_list_C.append(diff_C) if diff_C >= 0 else 0
            diff_list_G.append(diff_G) if diff_G >= 0 else 0

        diff_break = 10

        cut_off_A = int(sst.mode(np.around(np.divide(np.array(diff_list_A, dtype=np.float32), diff_break)))[0][0]) + \
                    diff_break / 2
        cut_off_T = int(sst.mode(np.around(np.divide(np.array(diff_list_T, dtype=np.float32), diff_break)))[0][0]) + \
                    diff_break / 2
        cut_off_C = int(sst.mode(np.around(np.divide(np.array(diff_list_C, dtype=np.float32), diff_break)))[0][0]) + \
                    diff_break / 2
        cut_off_G = int(sst.mode(np.around(np.divide(np.array(diff_list_G, dtype=np.float32), diff_break)))[0][0]) + \
                    diff_break / 2

        for key_point in mor_kps:

            r = int(key_point.pt[1])
            c = int(key_point.pt[0])

            if np.sum(self.__img_mat_A[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                    np.sum(self.__img_mat_A[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144 > cut_off_A:
                self.blob_greyscale_model_A[r, c] = \
                    np.sum(self.__img_mat_A[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                    np.sum(self.__img_mat_A[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144

            if np.sum(self.__img_mat_T[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                    np.sum(self.__img_mat_T[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144 > cut_off_T:
                self.blob_greyscale_model_T[r, c] = \
                    np.sum(self.__img_mat_T[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                    np.sum(self.__img_mat_T[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144

            if np.sum(self.__img_mat_C[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                    np.sum(self.__img_mat_C[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144 > cut_off_C:
                self.blob_greyscale_model_C[r, c] = \
                    np.sum(self.__img_mat_C[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                    np.sum(self.__img_mat_C[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144

            if np.sum(self.__img_mat_G[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                    np.sum(self.__img_mat_G[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144 > cut_off_G:
                self.blob_greyscale_model_G[r, c] = \
                    np.sum(self.__img_mat_G[(r - 2):(r + 4), (c - 2):(c + 4)]) / 36 - \
                    np.sum(self.__img_mat_G[(r - 5):(r + 7), (c - 5):(c + 7)]) / 144


########################
# Class of Base Caller #
########################

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


class BaseCaller:

    __base_pool = {}

    def __init__(self):

        self.bases_box = {}

    @classmethod
    def image_model_parsing(cls, f_image_model, f_base_type):

        for row in range(0, len(f_image_model)):
            for col in range(0, len(f_image_model[row])):

                read_id = 'r' + ('%04d' % (row + 1)) + 'c' + ('%04d' % (col + 1))

                if read_id not in BaseCaller.__base_pool:
                    BaseCaller.__base_pool.update({read_id: {'A': 0.0, 'T': 0.0, 'C': 0.0, 'G': 0.0}})

                if f_image_model[row, col] >= 0:
                    BaseCaller.__base_pool[read_id][f_base_type] = f_image_model[row, col]

    def mat2base(self):

        for read_id in BaseCaller.__base_pool:

            sorted_base_score = [base_score for base_score in
                                 sorted(BaseCaller.__base_pool[read_id].items(), key=lambda x: x[1], reverse=True)]

            if sorted_base_score[0][1] > sorted_base_score[1][1]:

                error_rate = np.around(sst.binom_test((sorted_base_score[0][1], sorted_base_score[1][1]),
                                                      p=0.5, alternative='greater'), 4)

                if read_id not in self.bases_box:
                    self.bases_box.update({read_id: [sorted_base_score[0][0], error_rate]})


class BasesCube:

    __bases_cube = []

    def __init__(self):

        self.adjusted_bases_cube = []

    @staticmethod
    def __check_greyscale(f_bases_cube, f_adjusted_bases_cube, f_cycle_id):

        f_adjusted_bases_cube[f_cycle_id] = {}

        for ref_coordinate in f_bases_cube[0]:

            min_qual_base = 'N'
            min_error_rate = 1.0000

            r = int(ref_coordinate[1:5].lstrip('0'))
            c = int(ref_coordinate[6:].lstrip('0'))

            for row in range(r - 2, r + 4):
                for col in range(c - 2, c + 4):

                    coor = str('r' + ('%04d' % row) + 'c' + ('%04d' % col))

                    if coor in f_bases_cube[f_cycle_id]:

                        error_rate = f_bases_cube[f_cycle_id][coor][1]

                        if error_rate < min_error_rate:

                            min_qual_base = f_bases_cube[f_cycle_id][coor][0]
                            min_error_rate = error_rate

            f_adjusted_bases_cube[f_cycle_id].update({ref_coordinate: [min_qual_base, min_error_rate]})

    @classmethod
    def collect_called_bases(cls, f_called_base_in_one_cycle):

        cls.__bases_cube.append(f_called_base_in_one_cycle)

    def calling_adjust(self):

        self.adjusted_bases_cube.append(self.__bases_cube[0])

        if len(self.__bases_cube) > 1:

            for cycle_id in range(1, len(self.__bases_cube)):
                self.adjusted_bases_cube.append({})

                self.__check_greyscale(self.__bases_cube, self.adjusted_bases_cube, cycle_id)

        else:
            print('There is only one cycle in this run', file=sys.stderr)


##################
# Basic Function #
##################

####################################################
# Function of Base Channel in Same Cycle Combining #
####################################################

def combine_base_channel_in_same_cycle(f_base_channel_img1,
                                       f_base_channel_img2,
                                       f_base_channel_img3,
                                       f_base_channel_img4,
                                       f_base_channel_img0):

    tmp_img1_2 = cv.addWeighted(f_base_channel_img1, 0.5,
                                f_base_channel_img2, 0.5, 0)
    tmp_img3_4 = cv.addWeighted(f_base_channel_img3, 0.5,
                                f_base_channel_img4, 0.5, 0)

    tmp_img1_2_3_4 = cv.addWeighted(tmp_img1_2, 0.5, tmp_img3_4, 0.5, 0)

    f_combined_image = cv.addWeighted(tmp_img1_2_3_4, 1, f_base_channel_img0, 1, 0)

    return f_combined_image


##################################
# Function of Image Registration #
##################################

def register_image(f_query_img, f_train_img):

    registered_image_obj = ImageRegistration(f_query_img, f_train_img)
    registered_image_obj.register_image()

    f_registered_image = registered_image_obj.registered_image
    f_matrix = registered_image_obj.matrix

    return f_registered_image, f_matrix


##################################
# Function of Image Registration #
##################################

def feature_detecting_and_modelization(f_image_matrix,
                                       f_image_matrix_A,
                                       f_image_matrix_T,
                                       f_image_matrix_C,
                                       f_image_matrix_G):

    f_detected_image_obj = Detector(f_image_matrix,
                                    f_image_matrix_A,
                                    f_image_matrix_T,
                                    f_image_matrix_C,
                                    f_image_matrix_G)

    f_detected_image_obj.blob_richly_detect()

    f_detected_image_A = f_detected_image_obj.blob_greyscale_model_A
    f_detected_image_T = f_detected_image_obj.blob_greyscale_model_T
    f_detected_image_C = f_detected_image_obj.blob_greyscale_model_C
    f_detected_image_G = f_detected_image_obj.blob_greyscale_model_G

    return f_detected_image_A, f_detected_image_T, f_detected_image_C, f_detected_image_G


############################
# Function of Base Calling #
############################

def base_calling(f_feature_detecting_imgModel1,
                 f_feature_detecting_imgModel2,
                 f_feature_detecting_imgModel3,
                 f_feature_detecting_imgModel4):

    calling_obj = BaseCaller()

    calling_obj.image_model_parsing(f_feature_detecting_imgModel1, 'A')
    calling_obj.image_model_parsing(f_feature_detecting_imgModel2, 'T')
    calling_obj.image_model_parsing(f_feature_detecting_imgModel3, 'C')
    calling_obj.image_model_parsing(f_feature_detecting_imgModel4, 'G')

    calling_obj.mat2base()

    calling_obj.__base_pool = None

    f_base_output = calling_obj.bases_box

    return f_base_output


#############################
# Function of Image File IO #
#############################

def write_reads_into_file(f_output_prefix, f_bases_cube):

    ou = open(f_output_prefix + '.txt', 'w')

    for j in f_bases_cube[0]:

        coo = [j[1:5], j[6:]]
        seq = []
        qul = []

        for k in range(0, len(sys.argv[1:])):

            if f_bases_cube[k][j][1] is not None:
                quality = int(-10 * np.log10(f_bases_cube[k][j][1] + 0.0001)) + 33

                seq.append(f_bases_cube[k][j][0])
                qul.append(chr(quality))

        print(j + '\t' + ''.join(seq) + '\t' + ''.join(qul) + '\t' + '\t'.join(coo), file=ou)

    ou.close()


#################
# Main Function #
#################

if __name__ == '__main__':
    """
        The main function is used to together all the other basic functions to generate a workflow.
        
        In our practice, this process is split into three main steps.
        
        Firstly, the channel images from same cycle are combined into one, after that, we make combined image from the 
        cycle 1 as the reference to register others, and record their registration matrices, respectively. Then, these 
        registration matrices from different registered cycles are used to translate and rotate their own 4 channels 
        images. All the images don't need scale due to their same focus.
        
        Next, a double strategy is employed to detect the fluorescent signal in each channel image, this strategy is 
        including a series of Morphological (TopHat) transformation and a Laplacian of Gaussian (LoG) transformation to 
        expose the high and low quality fluorescent signal as blobs, respectively. Then, a simple blob detection 
        algorithm taking a group of optimized parameters is invoked for locating.
        
        Lastly, the fluorescent signal is transformed into sequence information according to the order of the channel in
        each cycle.
    """

    bases_cube_obj = BasesCube()

    combined_image = []

    for i in range(0, len(sys.argv[1:])):

        channel_A_path = sys.argv[i + 1] + '/Y5.tif'
        channel_T_path = sys.argv[i + 1] + '/FAM.tif'
        channel_C_path = sys.argv[i + 1] + '/TXR.tif'
        channel_G_path = sys.argv[i + 1] + '/Y3.tif'
        channel_0_path = sys.argv[i + 1] + '/DAPI.tif'

        img_A = cv.imread(channel_A_path, cv.IMREAD_GRAYSCALE)
        img_T = cv.imread(channel_T_path, cv.IMREAD_GRAYSCALE)
        img_C = cv.imread(channel_C_path, cv.IMREAD_GRAYSCALE)
        img_G = cv.imread(channel_G_path, cv.IMREAD_GRAYSCALE)
        img_0 = cv.imread(channel_0_path, cv.IMREAD_GRAYSCALE)

        combined_image.append(combine_base_channel_in_same_cycle(img_A, img_T, img_C, img_G, img_0))

        registered_img, matrix = register_image(combined_image[0], combined_image[i])
        cv.imwrite('cycle_' + str(i + 1) + '.reg.tif', registered_img)  # debug

        registered_img_A = np.array([], dtype=np.uint8)
        registered_img_A = cv.warpAffine(img_A, matrix, (registered_img.shape[1], registered_img.shape[0]),
                                         registered_img_A, flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)

        registered_img_T = np.array([], dtype=np.uint8)
        registered_img_T = cv.warpAffine(img_T, matrix, (registered_img.shape[1], registered_img.shape[0]),
                                         registered_img_T, flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)

        registered_img_C = np.array([], dtype=np.uint8)
        registered_img_C = cv.warpAffine(img_C, matrix, (registered_img.shape[1], registered_img.shape[0]),
                                         registered_img_C, flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)

        registered_img_G = np.array([], dtype=np.uint8)
        registered_img_G = cv.warpAffine(img_G, matrix, (registered_img.shape[1], registered_img.shape[0]),
                                         registered_img_G, flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)

        (detected_registered_img_A,
         detected_registered_img_T,
         detected_registered_img_C,
         detected_registered_img_G) = feature_detecting_and_modelization(registered_img,
                                                                         registered_img_A,
                                                                         registered_img_T,
                                                                         registered_img_C,
                                                                         registered_img_G)

        called_base_list_in_one_cycle = base_calling(detected_registered_img_A,
                                                     detected_registered_img_T,
                                                     detected_registered_img_C,
                                                     detected_registered_img_G)

        bases_cube_obj.collect_called_bases(called_base_list_in_one_cycle)

    bases_cube_obj.calling_adjust()

    bases_cube = bases_cube_obj.adjusted_bases_cube

    write_reads_into_file('basecalling_data', bases_cube)
