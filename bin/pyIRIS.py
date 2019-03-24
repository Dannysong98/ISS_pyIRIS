#!/usr/bin/env python3


import sys

import cv2 as cv
import numpy as np


##################
# Data Structure #
##################


###########################
# Class ImageRegistration #
###########################

class ImageRegistration:

    def __init__(self, f_query_mat, f_train_mat):

        self.__query_mat = f_query_mat
        self.__train_mat = f_train_mat

        self.registered_image = np.array([], dtype='uint8')
        self.matrix = np.array([], dtype='uint8')

    @staticmethod
    def __get_kp(f_gray_img):

        detector = cv.BRISK_create()

        kp, des = detector.detectAndCompute(f_gray_img, None)

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

    def image_registration(self):

        kp1, des1 = self.__get_kp(self.__query_mat)
        kp2, des2 = self.__get_kp(self.__train_mat)

        good_match = self.__get_good_match(des1, des2)

        if len(good_match) > 4:

            pts_a = np.float32([kp1[_.queryIdx].pt for _ in good_match]).reshape(-1, 1, 2)
            pts_b = np.float32([kp2[_.trainIdx].pt for _ in good_match]).reshape(-1, 1, 2)

            self.matrix, _ = cv.findHomography(pts_a, pts_b, cv.RANSAC, 4)

            self.registered_image = cv.warpPerspective(self.__train_mat, self.matrix,
                                                       (self.__query_mat.shape[1], self.__query_mat.shape[0]),
                                                       flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)

        else:
            print('REGISTERING ERROR', file=sys.stderr)


##################
# Class Detector #
##################

class Detector:

    def __init__(self, f_img_taking_bg):

        self.__img_mat = f_img_taking_bg
        self.__contour_level = np.zeros(self.__img_mat.shape, dtype='uint8')

        self.keypoint_greyscale_model = np.zeros(self.__img_mat.shape, dtype='uint8')

    def detect(self):

        blur_img = cv.GaussianBlur(self.__img_mat, (5, 5), 0)

        lap = cv.Laplacian(blur_img, cv.CV_64F, ksize=3)
        dst = cv.convertScaleAbs(lap)

        detector = cv.BRISK_create()

        keypoints = detector.detect(dst)

        for keypoint in keypoints:

            r = int(keypoint.pt[1])
            c = int(keypoint.pt[0])

            self.__contour_level[(r - 1):(r + 3), (c - 1):(c + 3)] = self.__img_mat[(r - 1):(r + 3), (c - 1):(c + 3)]

        contours, _ = cv.findContours(self.__contour_level, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        for cnt in contours:

            if cv.contourArea(cnt) < 64:

                M = cv.moments(cnt)

                if M['m00'] > 0:

                    c_row = int(M['m01'] / M['m00'])
                    c_col = int(M['m10'] / M['m00'])

                    self.keypoint_greyscale_model[c_row, c_col] = \
                        np.sum(self.__img_mat[(c_row - 2):(c_row + 4), (c_col - 2):(c_col + 4)]) / 36
