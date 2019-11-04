#!/usr/bin/env python3
""""""


from sys import (argv, exit,
                 stderr)

from cv2 import (imread, createStitcherScans, cvtColor, imwrite, convertScaleAbs,
                 IMREAD_GRAYSCALE, IMREAD_COLOR, COLOR_BGR2GRAY)
from numpy import (array, dot, mean,
                   uint8, uint16)

from ..IRIS.register_images import register_cycles


def background_stitcher(img_dirs):
    """"""
    imgs = []

    for img_dir in img_dirs:
        imgs.append(imread(img_dir + '/background.tif', IMREAD_COLOR))

    for i in range(0, len(imgs)):
        imgs[i] = convertScaleAbs(imgs[i] * mean(imgs[0]) / mean(imgs[i]))

    stitcher = createStitcherScans()
    status, stitched_img = stitcher.stitch(imgs)

    if status == 0:
        stitched_img = cvtColor(stitched_img, COLOR_BGR2GRAY)
        return stitched_img

    else:
        print('FAIL TO STITCH', file=stderr)
        exit(1)


def trans_coor(bg, img_dirs):
    """"""
    adj_barcode_info = {}

    for img_dir in img_dirs:
        mat = register_cycles(bg, imread(img_dir + '/background.tif', IMREAD_GRAYSCALE), 'BRISK')

        with open(img_dir + '/basecalling_data.txt', 'rt') as IN:
            for ln in IN:
                ln = ln.split()

                seq = ln[1]
                qul = ln[2]

                row = int(ln[3].lstrip())
                col = int(ln[4].lstrip())

                col_row_tensor = array([col, row, 1], dtype=uint16)
                adj_col, adj_row = dot(mat, col_row_tensor)

                if adj_row < 0 or adj_col < 0:
                    continue

                adj_read_id = 'r%05d' % adj_row + 'c%05d' % adj_col

                adj_barcode_info.update({adj_read_id: '%s\t%s\t%05d\t%05d' % (seq, qul, adj_row, adj_col)})

    return adj_barcode_info


def overlap_filtering(adj_barcode_info):

    filtered_barcode_info = {}

    retained_keys_list = set([_ for _ in sorted(adj_barcode_info.keys())])

    for adj_cor_id in adj_barcode_info:

        seq, qul, row, col = adj_barcode_info[adj_cor_id].split()

        row = int(row)
        col = int(col)

        for r in range(row - 5, row + 7):
            for c in range(col - 5, col + 7):

                if r != row and c != col:

                    read_id = 'r%05dc%05d' % (r, c)

                    if read_id in retained_keys_list and \
                            seq == adj_barcode_info[read_id].split()[0] and \
                            qul == adj_barcode_info[read_id].split()[1]:
                        retained_keys_list.remove(read_id)

        if adj_cor_id in retained_keys_list and adj_cor_id not in filtered_barcode_info:
            filtered_barcode_info.update({adj_cor_id: adj_barcode_info[adj_cor_id]})

    return filtered_barcode_info


if __name__ == '__main__':
    stitched_image = array([], dtype=uint8)
    barcode_info = {}

    if argv[1] == '--bg':
        stitched_image = imread(argv[2], IMREAD_GRAYSCALE)
        barcode_info = overlap_filtering(trans_coor(stitched_image, argv[3:]))

    else:
        stitched_image = background_stitcher(argv[1:])
        barcode_info = overlap_filtering(trans_coor(stitched_image, argv[1:]))

    imwrite('all_background.tif', stitched_image)

    with open('all_basecalling_data.txt', 'wt') as OU:
        for b_info in barcode_info:
            print('\t'.join((b_info, barcode_info[b_info])), file=OU)
