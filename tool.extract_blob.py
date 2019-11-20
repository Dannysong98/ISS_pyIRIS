#!/usr/bin/env python3


import sys

import cv2 as cv
import numpy as np

grayList1 = []
grayList2 = []


def generate_stochastic_coordinate(f_imgFile, f_corFile):
    """"""
    img = cv.imread(f_imgFile, cv.IMREAD_GRAYSCALE)

    COR = open(f_corFile, 'rt')
    count = len(COR.readlines())
    COR.close()

    row_list = np.random.choice(range(0, img.shape[0]), count, replace=True)
    col_list = np.random.choice(range(0, img.shape[1]), count, replace=True)

    return row_list, col_list


def extract_blob_pixel(f_imgFile, f_corFile):
    """"""
    img1 = cv.imread(f_imgFile, cv.IMREAD_GRAYSCALE)
    img2 = img1.copy()

    diff_1 = 0
    diff_2 = 0

    with open(f_corFile) as COR:
        cor_list = COR.readlines()

        for _ in range(0, len(cor_list)):
            line = cor_list[_].split()

            if 'N' in line[1] or int(round(sum([ord(_) for _ in line[2]]) / len(line[2]))) < 0:
                continue

            r1 = int(line[3])
            c1 = int(line[4])

            r2 = generate_stochastic_coordinate(f_imgFile, f_corFile)[0][_]
            c2 = generate_stochastic_coordinate(f_imgFile, f_corFile)[1][_]

            diff_1 += np.sum(img1[(r1 - 1):(r1 + 3), (c1 - 1):(c1 + 3)]) / 16 - \
                      np.sum(img1[(r1 - 4):(r1 + 6), (c1 - 4):(c1 + 6)]) / 100

            diff_2 += np.sum(img2[(r2 - 1):(r2 + 3), (c2 - 1):(c2 + 3)]) / 16 - \
                      np.sum(img2[(r2 - 4):(r2 + 6), (c2 - 4):(c2 + 6)]) / 100

            grayList1.append((diff_1, diff_2))

            ########

            box_1 = np.sum(img1[(r1 - 1):(r1 + 3), (c1 - 1):(c1 + 3)]) / 16 - \
                      np.sum(img1[(r1 - 4):(r1 + 6), (c1 - 4):(c1 + 6)]) / 100

            box_2 = np.sum(img2[(r2 - 1):(r2 + 3), (c2 - 1):(c2 + 3)]) / 16 - \
                      np.sum(img2[(r2 - 4):(r2 + 6), (c2 - 4):(c2 + 6)]) / 100

            grayList2.append((box_1, box_2))

            ########

            img1[(r1 - 1):(r1 + 3), (c1 - 1):(c1 + 3)] = 255
            img2[(r2 - 1):(r2 + 3), (c2 - 1):(c2 + 3)] = 0

    cv.imwrite('debug.deteDel.tif', img1)
    cv.imwrite('debug.stocDel.tif', img2)

    with open('debug.dete_stoc.ext.txt', 'w') as OU:
        for _ in grayList1:
            print('%d\tdete' % (_[0]), file=OU)
            print('%d\tstoc' % (_[1]), file=OU)

    with open('debug.dete_stoc.box.txt', 'w') as OU:
        for _ in grayList2:
            print('%d\tdete' % (_[0]), file=OU)
            print('%d\tstoc' % (_[1]), file=OU)


if __name__ == '__main__':
    extract_blob_pixel(sys.argv[1], sys.argv[2])
