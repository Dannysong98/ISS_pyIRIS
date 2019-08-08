#!/usr/bin/env python3
""""""


from cv2 import imwrite
from numpy import log10


def write_reads_into_file(f_background, f_barcode_cube, f_cycle_num):
    """"""
    imwrite('background.tif', f_background)

    ou = open('basecalling_data.txt', 'w')

    for j in f_barcode_cube[0]:
        coo = [j[1:6], j[7:]]
        seq = []
        qul = []

        for k in range(0, f_cycle_num):
            if f_barcode_cube[k][j][1] is not None:
                # Transforming the Error Rate into the Phred+ 33 Score #
                quality = int(-10 * log10(f_barcode_cube[k][j][1] + 0.0001)) + 33

                seq.append(f_barcode_cube[k][j][0])
                qul.append(chr(quality))

        print(j + '\t' + ''.join(seq) + '\t' + ''.join(qul) + '\t' + '\t'.join(coo), file=ou)

    ou.close()
