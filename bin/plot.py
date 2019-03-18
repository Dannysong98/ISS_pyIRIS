#!/usr/bin/env python3


import sys

import cv2 as cv

if __name__ == '__main__':

    bg_imgMat = cv.imread(sys.argv[2])

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

    for coordinate in calling_box[sys.argv[3]]:
        cv.circle(bg_imgMat, (coordinate[1] - 1, coordinate[0] - 1), 4, (255, 128, 128), 1)

    for coordinate in calling_box[sys.argv[4]]:
        cv.circle(bg_imgMat, (coordinate[1] - 1, coordinate[0] - 1), 4, (128, 255, 128), 1)

    for coordinate in calling_box[sys.argv[5]]:
        cv.circle(bg_imgMat, (coordinate[1] - 1, coordinate[0] - 1), 4, (128, 128, 255), 1)

    for coordinate in calling_box[sys.argv[6]]:
        cv.circle(bg_imgMat, (coordinate[1] - 1, coordinate[0] - 1), 4, (255, 255, 128), 1)

    cv.imwrite('plot.' + sys.argv[3] + '_' + sys.argv[4] + '_' + sys.argv[5] + '_' + sys.argv[6] + '.tif', bg_imgMat)
