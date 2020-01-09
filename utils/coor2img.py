#!/usr/bin/env python3


import sys
import cv2 as cv
import numpy as np


def generate_img(f_coor, f_bg, mark=None):
    bg_mat = cv.imread(f_bg, cv.IMREAD_COLOR)

    with open(f_coor) as FH:
        for i in FH:
            if '#' in i[0]:
                continue

            i = i.split()

            if mark == '--noN':
                if 'N' in i[1]:
                    continue

            row = int(np.around(float(i[3]),))
            col = int(np.around(float(i[4]),))

            img = cv.circle(bg_mat, (col, row), 0, (0, 255, 0), 2)

    cv.imwrite(f_bg + '.debug.tif', img)


if __name__ == '__main__':
    if len(sys.argv) == 3:
        generate_img(sys.argv[1], sys.argv[2])

    else:
        generate_img(sys.argv[1], sys.argv[2], sys.argv[3])
