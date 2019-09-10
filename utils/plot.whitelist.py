#!/usr/bin/env python3


import sys
import os.path

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib import rcParams


if __name__ == '__main__':

    code_box = {}

    with open(sys.argv[1]) as FH:

        for line in FH:

            line = line.split()

            row = int(line[3])
            col = int(line[4])

            if 'N' in line[1]:
                continue

            if line[1] not in code_box:
                code_box.update({line[1]: []})

            code_box[line[1]].append([row, col])

    img = cv.imread(sys.argv[2])

    rcParams['font.family'] = 'monospace'

    fig = plt.figure(figsize=(img.shape[1] / 100, img.shape[0] / 100))

    fig.add_subplot(1, 1, 1, label='p1.1',
                    xlim=(0, img.shape[1] + 1), ylim=(img.shape[0], 1),
                    xticks=[], yticks=[], frame_on=False)

    plt.imshow(img)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    fig.add_subplot(1, 1, 1, label='p1.2',
                    xlim=(0, img.shape[1] + 1), ylim=(-img.shape[0], 1),
                    xticks=[], yticks=[], frame_on=False)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    whitelist = []

    col_list = ['#e31a1c',
                '#ff7f00',
                '#33a02c',
                '#1f78b4',
                '#6a3d9a',
                '#fb9a99',
                '#fdbf6f',
                '#c3df8a',
                '#b6cee3',
                '#bab2d6',
                '#f31afc',
                '#fa7f00',
                '#32a02c',
                '#117fb4',
                '#6b3d9a',
                '#fcfa99',
                '#ffbf6f',
                '#b3df8a',
                '#a6fee3',
                '#c0e2d6',
                '#e21aee',
                '#fe5e00',
                '#32b02c',
                '#124fb4',
                '#7a3d9a',
                '#fdea99',
                '#ffff6f',
                '#b3df8a',
                '#a7fee3',
                '#cfe3d6',
                '#e11a1c',
                '#ef7f00',
                '#f3a02c',
                '#ef78b4',
                '#4a3d9a',
                '#db9a99',
                '#cdbf6f',
                '#a3df8a',
                '#becee3',
                '#bdb2d6']

    if len(sys.argv) > 3:

        for k in sys.argv[3:-1]:
            whitelist.append(k)

    else:
        print('Please input the barcode you want to show', file=sys.stderr)

    content = []

    n = 0

    for k in whitelist:

        i, j = k.split(',')

        if i in code_box:

            plt.scatter(np.asarray(code_box[i])[:, 1] - 1, -np.asarray(code_box[i])[:, 0] + 1,
                        s=300, label=i, marker='o', c='', edgecolor=col_list[n], alpha=1, linewidths=5)

            content.append('%6d %s (%s) %s' % (len(code_box[i]), i, j, col_list[n]))

        n += 1

    if int(sys.argv[-1]) == 1:
        plt.legend(content, loc=2, markerscale=2, fontsize=32)

    out_prefix = os.path.basename(sys.argv[1])

    if len(sys.argv) >= 4:
        out_prefix = sys.argv[3].replace(',', '_')

    plt.savefig(out_prefix + '.exp_location.tif', format='tif', ppi=300)
