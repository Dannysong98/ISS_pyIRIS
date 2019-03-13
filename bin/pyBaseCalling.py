#!/usr/bin/env python3


"""

    Author: Hao Yu (yuhao@genomics.cn)
            Yang Zhou (zhouyang@genomics.cn)

    Initial Date:   2018-10-21

    ChangeLog:
        v1.0
            2019-01-15
                First release

"""


import re
import sys

import cv2 as cv
import numpy as np


##################
# Data Structure #
##################

####################
# Class of Profile #
####################

class Profile:

    def __init__(self, f_profile_path):

        self.__profile_path = f_profile_path
        self.__par_box = {'channel_A': 'Y5.tif', 'channel_T': 'GFP.tif', 'channel_C': 'TXR.tif', 'channel_G': 'Y3.tif',
                          'cycle_mask': ['#1', '#2', '#3'],
                          'root_path': './'}

        self.par_pool = {'baseImg_path_A': [], 'baseImg_path_T': [], 'baseImg_path_C': [], 'baseImg_path_G': []}

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

    def __init__(self, f_query_mat, f_train_mat):

        self.__query_mat = f_query_mat
        self.__train_mat = f_train_mat

        self.registered_image = np.array([], dtype='uint8')
        self.matrix = np.zeros((3, 3), dtype='float32')

    @staticmethod
    def __get_kp(f_gray_img):

        detector = cv.BRISK_create()

        kp, des = detector.detectAndCompute(f_gray_img, None)

        return kp, des

    @staticmethod
    def __get_good_match(f_des1, f_des2):

        matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

        matches = matcher.knnMatch(f_des1, f_des2, 2)

        good_matches = []

        for m, n in matches:

            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        good_matches = sorted(good_matches, key=lambda x: x.distance)

        return good_matches

    def image_registration(self):

        kp1, des1 = self.__get_kp(self.__query_mat)
        kp2, des2 = self.__get_kp(self.__train_mat)

        good_matchs = self.__get_good_match(des1, des2)

        if len(good_matchs) > 4:

            pts_a = np.float32([kp1[_.queryIdx].pt for _ in good_matchs]).reshape(-1, 1, 2)
            pts_b = np.float32([kp2[_.trainIdx].pt for _ in good_matchs]).reshape(-1, 1, 2)

            self.matrix, _ = cv.findHomography(pts_a, pts_b, cv.RANSAC, 4)

            self.registered_image = cv.warpPerspective(self.__train_mat, self.matrix,
                                                       (self.__query_mat.shape[1], self.__query_mat.shape[0]),
                                                       self.registered_image,
                                                       flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)

        else:
            print('REGISTERING ERROR', file=sys.stderr)


#############################
# Class of Feature Detector #
#############################

class Detector:

    def __init__(self, f_img_taking_bg):

        self.__img_mat = f_img_taking_bg
        self.__contour_level = np.zeros(self.__img_mat.shape, dtype='uint8')

        self.keypoint_greyscale_model = np.zeros(self.__img_mat.shape, dtype='uint8')

    def detect(self):

        blur_img = np.zeros(self.__img_mat.shape, dtype='uint8')
        blur_img = cv.GaussianBlur(self.__img_mat, (3, 3), 1, blur_img)

        lap1 = np.zeros(self.__img_mat.shape, dtype='uint8')
        lap2 = np.zeros(self.__img_mat.shape, dtype='uint8')

        lap1 = cv.Laplacian(blur_img, cv.CV_16S, lap1, 1)
        lap2 = cv.Laplacian(blur_img, cv.CV_16S, lap2, 3)

        lap1 = cv.convertScaleAbs(lap1)
        lap2 = cv.convertScaleAbs(lap2)

        lap_img = cv.absdiff(lap1, lap2)

        detector = cv.BRISK_create()

        keypoints = detector.detect(lap_img)

        for keypoint in keypoints:

            r = int(keypoint.pt[1])
            c = int(keypoint.pt[0])

            self.__contour_level[r:(r + 2), c:(c + 2)] = self.__img_mat[r:(r + 2), c:(c + 2)]

        _, contours, _ = cv.findContours(self.__contour_level, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

        for cnt in contours:

            if cv.contourArea(cnt) < 64:

                M = cv.moments(cnt)

                if M['m00'] > 0:

                    c_row = int(M['m01'] / M['m00'])
                    c_col = int(M['m10'] / M['m00'])

                    self.keypoint_greyscale_model[c_row, c_col] = \
                        np.sum(self.__img_mat[(c_row - 2):(c_row + 4), (c_col - 2):(c_col + 4)]) / 36


########################
# Class of Base Caller #
########################

class BaseCaller:

    __base_pool = {}

    def __init__(self):

        self.bases_box = {}

    @classmethod
    def image_model_parsing(cls, f_image_model, f_base_type):

        for row in range(0, len(f_image_model) - 1):
            for col in range(0, len(f_image_model[row]) - 1):

                read_id = 'r' + ('%04d' % (row + 1)) + 'c' + ('%04d' % (col + 1))

                if read_id not in cls.__base_pool:
                    cls.__base_pool.update({read_id: {'A': 0.0, 'T': 0.0, 'C': 0.0, 'G': 0.0}})

                cls.__base_pool[read_id][f_base_type] = f_image_model[row, col]

    def mat2base(self):

        for n in BaseCaller.__base_pool:

            if n not in self.bases_box:
                self.bases_box.update({n: []})

            sorted_base_quality = [base_quality for base_quality in
                                   sorted(BaseCaller.__base_pool[n].items(), key=lambda x: x[1], reverse=True)]

            quality_sum = sum([sorted_base_quality[0][1], sorted_base_quality[1][1],
                               sorted_base_quality[2][1], sorted_base_quality[3][1]])

            if quality_sum > 0:

                # quality = int(-10 * np.log10(1 - sorted_base_quality[0][1] / quality_sum * 0.9999)) + 33
                quality = int(-10 * np.log10(1 - sorted_base_quality[0][1] / 255 * 0.9999)) + 33

                if quality > 33:

                    self.bases_box[n].append(sorted_base_quality[0][0])
                    self.bases_box[n].append(chr(quality))

                else:

                    self.bases_box[n].append(sorted_base_quality[0][0])
                    self.bases_box[n].append(chr(34))

            else:

                self.bases_box[n].append('N')
                self.bases_box[n].append(chr(33))


class BasesCube:

    __bases_cube = []

    def __init__(self):

        self.adjusted_bases_cube = []

    @staticmethod
    def __check_greyscale(f_bases_cube, f_adjusted_bases_cube, f_cycle_id):

        for ref_coordinate in f_bases_cube[0]:

            if f_bases_cube[0][ref_coordinate][0] != 'N':

                max_qual_base = 'N'
                max_qual = 33

                r = int(ref_coordinate[1:5].lstrip('0'))
                c = int(ref_coordinate[6:].lstrip('0'))

                for row in range(r - 8, r + 10):
                    for col in range(c - 8, c + 10):

                        # m_dis = abs(row - r) + abs(col - c)

                        # MEAN, SD, MAG = (0, 5, 12.5)
                        # W = np.exp(-((m_dis - MEAN) ** 2)/(2 * SD ** 2)) / (SD * np.sqrt(2 * np.pi)) * MAG

                        coor = str('r' + ('%04d' % row) + 'c' + ('%04d' % col))

                        if coor in f_bases_cube[0]:

                            # if int(ord(f_bases_cube[f_cycle_id][coor][1]) * W) > max_qual:
                            if int(ord(f_bases_cube[f_cycle_id][coor][1])) > max_qual:

                                max_qual_base = f_bases_cube[f_cycle_id][coor][0]
                                max_qual = int(ord(f_bases_cube[f_cycle_id][coor][1]))

                if max_qual == 33 and max_qual_base != 'N':
                    max_qual = 34

                if ref_coordinate not in f_adjusted_bases_cube[f_cycle_id]:
                    f_adjusted_bases_cube[f_cycle_id].update({ref_coordinate: [max_qual_base, chr(max_qual)]})

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

def combine_base_channel_in_same_cycle(f_base_channel_img1, f_base_channel_img2,
                                       f_base_channel_img3, f_base_channel_img4):

    tmp_img1_2 = cv.addWeighted(f_base_channel_img1, 0.8, f_base_channel_img2, 0.8, 0)
    tmp_img3_4 = cv.addWeighted(f_base_channel_img3, 0.8, f_base_channel_img4, 0.8, 0)

    f_combined_image = cv.addWeighted(tmp_img1_2, 0.8, tmp_img3_4, 0.8, 0)

    return f_combined_image


##################################
# Function of Image Registration #
##################################

def register_image(f_query_img, f_train_img):

    registered_image_obj = ImageRegistration(f_query_img, f_train_img)
    registered_image_obj.image_registration()

    f_registered_image = registered_image_obj.registered_image
    f_matrix = registered_image_obj.matrix

    return f_registered_image, f_matrix


##################################
# Function of Image Registration #
##################################

def feature_detecting_and_modelization(f_image_matrix):

    f_detected_image_obj = Detector(f_image_matrix)
    f_detected_image_obj.detect()
    f_detected_image = f_detected_image_obj.keypoint_greyscale_model

    return f_detected_image


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

    f_base_output = calling_obj.bases_box

    return f_base_output


#############################
# Function of Image File IO #
#############################

def write_reads_into_file(f_output_prefix, f_bases_cube):

    ou = open(f_output_prefix + '.lst.txt', 'w')

    for j in f_bases_cube[0]:

        if f_bases_cube[0][j][0] != 'N':

            seq = []
            qul = []

            for k in range(0, len(sys.argv[1:])):

                seq.append(f_bases_cube[k][j][0])
                qul.append(f_bases_cube[k][j][1])

            print(j + '\t' + ''.join(seq) + '\t' + ''.join(qul), file=ou)

    ou.close()


#################
# Main Function #
#################

if __name__ == '__main__':

    bases_cube_obj = BasesCube()

    combined_image = []

    for i in range(0, len(sys.argv[1:])):

        channel_A_path = sys.argv[i + 1] + '/Y5.tif'
        channel_T_path = sys.argv[i + 1] + '/FAM.tif'
        channel_C_path = sys.argv[i + 1] + '/TXR.tif'
        channel_G_path = sys.argv[i + 1] + '/Y3.tif'

        img_A = cv.imread(channel_A_path, cv.IMREAD_GRAYSCALE)
        img_T = cv.imread(channel_T_path, cv.IMREAD_GRAYSCALE)
        img_C = cv.imread(channel_C_path, cv.IMREAD_GRAYSCALE)
        img_G = cv.imread(channel_G_path, cv.IMREAD_GRAYSCALE)

        combined_image.append(combine_base_channel_in_same_cycle(img_A, img_T, img_C, img_G))
        # cv.imwrite('cycle_' + str(i + 1) + '.com.tif', combined_image[i])  # debug

        registered_img, matrix = register_image(combined_image[0], combined_image[i])
        # cv.imwrite('cycle_' + str(i + 1) + '.reg.tif', registered_img)  # debug

        registered_img_A = np.array([], dtype='uint8')
        registered_img_A = cv.warpPerspective(img_A, matrix, (registered_img.shape[1], registered_img.shape[0]),
                                              registered_img_A, flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)

        registered_img_T = np.array([], dtype='uint8')
        registered_img_T = cv.warpPerspective(img_T, matrix, (registered_img.shape[1], registered_img.shape[0]),
                                              registered_img_T, flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)

        registered_img_C = np.array([], dtype='uint8')
        registered_img_C = cv.warpPerspective(img_C, matrix, (registered_img.shape[1], registered_img.shape[0]),
                                              registered_img_C, flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)

        registered_img_G = np.array([], dtype='uint8')
        registered_img_G = cv.warpPerspective(img_G, matrix, (registered_img.shape[1], registered_img.shape[0]),
                                              registered_img_G, flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)

        detected_registered_img_A = feature_detecting_and_modelization(registered_img_A)
        detected_registered_img_T = feature_detecting_and_modelization(registered_img_T)
        detected_registered_img_C = feature_detecting_and_modelization(registered_img_C)
        detected_registered_img_G = feature_detecting_and_modelization(registered_img_G)

        called_base_list_in_one_cycle = base_calling(detected_registered_img_A,
                                                     detected_registered_img_T,
                                                     detected_registered_img_C,
                                                     detected_registered_img_G)

        bases_cube_obj.collect_called_bases(called_base_list_in_one_cycle)

    bases_cube_obj.calling_adjust()

    bases_cube = bases_cube_obj.adjusted_bases_cube

    write_reads_into_file('reads.output', bases_cube)
