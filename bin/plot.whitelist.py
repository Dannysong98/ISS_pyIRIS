#!/usr/bin/env python3


import sys

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


if __name__ == '__main__':

    code_box = {}

    with open(sys.argv[1]) as FH:

        for line in FH:

            line = line.split()

            row = int(line[0][1:5])
            col = int(line[0][6:])

            if 'N' in line[1]:
                continue

            if line[1] not in code_box:
                code_box.update({line[1]: []})

            code_box[line[1]].append([row, col])

    img = cv.imread(sys.argv[2])

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

    n = 0

    whitelist = []

    col_list = ['#e31a1c', '#fb9a99',
                '#ff7f00', '#fdbf6f',
                '#33a02c', '#b2df8a',
                '#1f78b4', '#a6cee3',
                '#6a3d9a', '#cab2d6']

    content = []

    for i in sorted(code_box.keys(), key=lambda x: len(code_box[x]), reverse=True)[:10]:

        if len(sys.argv) > 3:

            if i in sys.argv[3:]:
                whitelist.append(i)

        else:
            whitelist.append(i)

    for i in whitelist:

        plt.scatter(np.asarray(code_box[i])[:, 1] - 1, -np.asarray(code_box[i])[:, 0] + 1,
                    s=12, label=i, marker='o', c='', edgecolor=col_list[n], alpha=1)

        content.append('%s: %i' % (i, np.shape(code_box[i])[0]))

        n += 1

    plt.legend(content, loc=0, markerscale=4)

    if len(sys.argv) <= 3:
        plt.savefig(sys.argv[1] + '.TOP10_BARCODE.eps', format='eps', ppi=300)

    else:
        plt.savefig(sys.argv[1] + '.' + '_'.join(sys.argv[3:]) + '.eps', format='eps', ppi=300)
