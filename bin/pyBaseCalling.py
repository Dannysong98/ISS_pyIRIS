#!/usr/bin/env python3


"""

    Author: Hao Yu  (yuhao@genomics.cn)
    Date:   2018-10-21

    ChangeLog:
        v1.0
            2019-01-15
                First release

"""


import sys

import cv2 as cv
import numpy as np


######################
# Initiating classes #
######################


class ImageRegistration:

    def __init__(self, f_query_mat, f_train_mat):

        self.__query_mat = f_query_mat
        self.__train_mat = f_train_mat

        self.registered_image = np.array([], dtype='uint8')

    @staticmethod
    def __sift_kp(f_gray_img):

        sift = cv.xfeatures2d.SIFT_create()

        kp, des = sift.detectAndCompute(f_gray_img, None)

        return kp, des

    @staticmethod
    def __get_good_match(f_des1, f_des2):

        bf = cv.BFMatcher()

        matches = bf.knnMatch(f_des1, f_des2, k=2)

        good_matches = []

        for m, n in matches:

            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        return good_matches

    def sift_image_registration(self):

        kp1, des1 = self.__sift_kp(self.__query_mat)
        kp2, des2 = self.__sift_kp(self.__train_mat)

        good_match = self.__get_good_match(des1, des2)

        if len(good_match) > 4:

            pts_a = np.float32([kp1[_.queryIdx].pt for _ in good_match]).reshape(-1, 1, 2)
            pts_b = np.float32([kp2[_.trainIdx].pt for _ in good_match]).reshape(-1, 1, 2)

            h, _ = cv.findHomography(pts_a, pts_b, cv.RANSAC, 4)

            self.registered_image = cv.warpPerspective(self.__train_mat, h,
                                                       (self.__query_mat.shape[1],
                                                        self.__query_mat.shape[0]),
                                                       flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)


class Thresholding:

    def __init__(self, f_image_matrix):

        self.__image_matrix = f_image_matrix
        self.__ksize = 0
        self.__sigma = 0

        self.thresholding_image = np.array([], dtype='uint8')

    def get_gaussian_kernel(self, f_block_diameter):

        self.__ksize = int(f_block_diameter / 0.95 * 0.68) \
            if int(f_block_diameter / 0.95 * 0.68) % 2 == 1 else int(f_block_diameter / 0.95 * 0.68) + 1

        self.__sigma = 0.3 * ((self.__ksize - 1) * 0.5 - 1) + 0.8

    def thresholding_by_otsu(self):

        blured_image = cv.GaussianBlur(self.__image_matrix, (self.__ksize, self.__ksize), self.__sigma, 0,
                                       cv.BORDER_CONSTANT)

        _, self.thresholding_image = cv.threshold(blured_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    def auto_thresholding(self):

        self.thresholding_image = cv.adaptiveThreshold(self.__image_matrix, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                       cv.THRESH_BINARY, self.__ksize, 2)


class ImageModel:

    def __init__(self, f_thresholding_image_matrix):

        self.__image_matrix = f_thresholding_image_matrix

        self.modelled_image = np.array([], dtype='uint8')

    def simple_block_modeling(self, f_edge_length):

        grayscale_box = []

        maximum_row, maximum_col = self.__image_matrix.shape

        n = 0

        for row in range(0, maximum_row - f_edge_length, f_edge_length):
            grayscale_box.append([])

            for col in range(0, maximum_col - f_edge_length, f_edge_length):
                block_total_grayscale = 0

                for sub_row in range(row, row + f_edge_length):
                    for sub_col in range(col, col + f_edge_length):
                        block_total_grayscale += self.__image_matrix[sub_row][sub_col]

                grayscale_box[n].append(float(block_total_grayscale) / (f_edge_length ** 2))

            n += 1

        self.modelled_image = np.array(grayscale_box, dtype='uint8')


class Filter:

    def __init__(self, f_modelled_image_matrix):

        self.__image_matrix = f_modelled_image_matrix

        self.filtered_image = np.zeros((f_modelled_image_matrix.shape[0],
                                        f_modelled_image_matrix.shape[1]),
                                       dtype='uint8')

    def filter_by_contour_size(self, f_size_limit):

        _, contour_list, _ = cv.findContours(self.__image_matrix, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        for contour in contour_list:

            if 1 <= len(contour) <= f_size_limit:
                for dot_list in contour:

                    for dot in dot_list:
                        self.filtered_image[dot[1]][dot[0]] = self.__image_matrix[dot[1]][dot[0]]


class BaseCaller:

    base_mat = {}

    def __init__(self):

        self.bases_box = {}

    @classmethod
    def image_model_parsing(cls, f_image_model, f_base_type):

        for row in range(0, len(f_image_model) - 1):
            for col in range(0, len(f_image_model[row]) - 1):

                read_id = 'r' + ('%04d' % (row + 1)) + 'c' + ('%04d' % (col + 1))

                if read_id not in cls.base_mat:
                    cls.base_mat.update({read_id: {'A': 0.0, 'T': 0.0, 'C': 0.0, 'G': 0.0}})

                cls.base_mat[read_id][f_base_type] = f_image_model[row][col]

    def mat2base(self):

        for n in BaseCaller.base_mat:

            if n not in self.bases_box:
                self.bases_box.update({n: []})

            sorted_base_quality = [base_quality for base_quality in
                                   sorted(BaseCaller.base_mat[n].items(), key=lambda x: x[1], reverse=True)]

            quality_sum = sum([sorted_base_quality[0][1], sorted_base_quality[1][1],
                               sorted_base_quality[2][1], sorted_base_quality[3][1]])

            if quality_sum > 0:

                self.bases_box[n].append(sorted_base_quality[0][0])

                # Get sequencing quality #
                self.bases_box[n].append(chr(
                    int(-10 * np.log10(1 - (float(sorted_base_quality[0][1]) / float(quality_sum)) * 0.9999)) + 33
                ))
                ##########################

            else:

                self.bases_box[n].append('N')
                self.bases_box[n].append(chr(33))


############
# Function #
############


def combine_base_channel_in_same_cycle(f_base_channel_img1, f_base_channel_img2,
                                       f_base_channel_img3, f_base_channel_img4):

    tmp_img1_2 = cv.addWeighted(f_base_channel_img1, 1, f_base_channel_img2, 1, 0)
    tmp_img3_4 = cv.addWeighted(f_base_channel_img3, 1, f_base_channel_img4, 1, 0)

    f_combined_image = cv.addWeighted(tmp_img1_2, 1, tmp_img3_4, 1, 0)

    return f_combined_image


def register_image(f_query_img, f_train_img):

    registered_image_obj = ImageRegistration(f_query_img, f_train_img)
    registered_image_obj.sift_image_registration()

    f_registered_image = registered_image_obj.registered_image

    return f_registered_image


def thresholding_modelization_filtering(f_image_matrix, f_method=None):

    f_thresholding_image_obj = Thresholding(f_image_matrix)
    f_thresholding_image_obj.get_gaussian_kernel(6)

    if f_method is not None and f_method == 'otsu':
        f_thresholding_image_obj.thresholding_by_otsu()

    else:
        f_thresholding_image_obj.auto_thresholding()

    f_thresholding_image = f_thresholding_image_obj.thresholding_image

    f_modelled_image_obj = ImageModel(f_thresholding_image)
    f_modelled_image_obj.simple_block_modeling(6)
    f_modelled_image = f_modelled_image_obj.modelled_image

    f_filtered_image_obj = Filter(f_modelled_image)
    f_filtered_image_obj.filter_by_contour_size(9)
    f_filtered_image = f_filtered_image_obj.filtered_image

    return f_filtered_image


def base_calling(f_filtered_registered_base_img1, f_filtered_registered_base_img2,
                 f_filtered_registered_base_img3, f_filtered_registered_base_img4):

    calling_obj = BaseCaller()

    calling_obj.image_model_parsing(f_filtered_registered_base_img1, 'A')
    calling_obj.image_model_parsing(f_filtered_registered_base_img2, 'T')
    calling_obj.image_model_parsing(f_filtered_registered_base_img3, 'C')
    calling_obj.image_model_parsing(f_filtered_registered_base_img4, 'G')

    calling_obj.mat2base()

    f_base_output = calling_obj.bases_box

    return f_base_output


def write_reads_into_file(f_output, f_bases_cube):

    ou = open(f_output, 'w')

    for j in f_bases_cube[0]:

        seq = []
        qul = []

        for k in range(0, len(sys.argv[1:])):

            seq.append(f_bases_cube[k][j][0])
            qul.append(f_bases_cube[k][j][1])

        print(j + '\t' + ''.join(seq) + '\t' + ''.join(qul), file=ou)

    ou.close()


########
# main #
########


if __name__ == '__main__':

    bases_cube = []

    combined_image = []

    for i in range(0, len(sys.argv[1:])):

        channel_A_path = sys.argv[i + 1] + '/Y5.tif'
        channel_T_path = sys.argv[i + 1] + '/GFP.tif'
        channel_C_path = sys.argv[i + 1] + '/TXR.tif'
        channel_G_path = sys.argv[i + 1] + '/Y3.tif'

        img_A = cv.imread(channel_A_path, cv.IMREAD_GRAYSCALE)
        img_T = cv.imread(channel_T_path, cv.IMREAD_GRAYSCALE)
        img_C = cv.imread(channel_C_path, cv.IMREAD_GRAYSCALE)
        img_G = cv.imread(channel_G_path, cv.IMREAD_GRAYSCALE)

        combined_image.append(combine_base_channel_in_same_cycle(img_A, img_T, img_C, img_G))

        registered_img = register_image(combined_image[0], combined_image[i])

        registered_img_A = register_image(registered_img, img_A)
        registered_img_T = register_image(registered_img, img_T)
        registered_img_C = register_image(registered_img, img_C)
        registered_img_G = register_image(registered_img, img_G)

        filtered_img_A = thresholding_modelization_filtering(registered_img_A, 'otsu')
        filtered_img_T = thresholding_modelization_filtering(registered_img_T, 'otsu')
        filtered_img_C = thresholding_modelization_filtering(registered_img_C, 'otsu')
        filtered_img_G = thresholding_modelization_filtering(registered_img_G, 'otsu')

        called_base = base_calling(filtered_img_A, filtered_img_T, filtered_img_C, filtered_img_G)

        bases_cube.append(called_base)

    write_reads_into_file('reads.output.txt', bases_cube)
