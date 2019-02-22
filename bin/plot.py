#!/usr/bin/env python3


import sys

import cv2 as cv

if __name__ == '__main__':

    bg_imgMat = cv.imread('1/DAPI.tif')

    calling_box = {}

    with open(sys.argv[1], 'r') as FH:

        for ln in FH:

            ln = ln.split()

            code = ln[1]

            r = int(ln[0][1:5].lstrip('0'))
            c = int(ln[0][6:].lstrip('0'))

            coor = (r, c)

            if code not in calling_box:
                calling_box.update({code: []})

            calling_box[code].append(coor)

    for coordinate in calling_box[sys.argv[2]]:
        cv.circle(bg_imgMat, (coordinate[1] - 1, coordinate[0] - 1), 4, (247, 80, 166), 1)

    for coordinate in calling_box[sys.argv[3]]:
        cv.circle(bg_imgMat, (coordinate[1] - 1, coordinate[0] - 1), 4, (166, 247, 255), 1)

    cv.imwrite('plot.' + sys.argv[2] + '_' + sys.argv[3] + '.tif', bg_imgMat)
