#!/usr/bin/env python3


import sys

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.colors as clr

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

    fig = plt.figure(figsize=(img.shape[1]/100, img.shape[0]/100))

    fig.add_subplot(1, 1, 1, label='p1.1',
                    xlim=(0, img.shape[1] + 1), ylim=(img.shape[0], 1),
                    xticks=[], yticks=[], frame_on=False)

    plt.imshow(img)

    fig.add_subplot(1, 1, 1, label='p1.2',
                    xlim=(0, img.shape[1] + 1), ylim=(-img.shape[0], 1),
                    xticks=[], yticks=[], frame_on=False)

    n = 0
    content = []

    if len(sys.argv) <= 3:

        for i in sorted(code_box.keys(), key=lambda x: len(code_box[x]), reverse=True):

            plt.scatter(np.asarray(code_box[i])[:, 1] - 1, -np.asarray(code_box[i])[:, 0] + 1,
                        s=9, label=i, marker='o', c=[c for c in clr.cnames.values()][n], edgecolor='', alpha=1)

            content.append('%s: %i' % (i, np.shape(code_box[i])[0]))

            n += 1

        plt.legend(content, loc=0, markerscale=4)

        plt.savefig(sys.argv[1] + '.ALL_BARCODE.eps', format='eps', ppi=300)

    else:

        for i in sorted(code_box.keys(), key=lambda x: len(code_box[x]), reverse=True):

            if '-' + i in sys.argv[3:]:
                continue

            plt.scatter(np.asarray(code_box[i])[:, 1] - 1, -np.asarray(code_box[i])[:, 0] + 1,
                        s=9, label=i, marker='o', c=[c for c in clr.cnames.values()][n], edgecolor='', alpha=1)

            content.append('%s: %i' % (i, np.shape(code_box[i])[0]))

            n += 1

        plt.legend(content, loc=0, markerscale=4)

        plt.savefig(sys.argv[1] + '.' + '_'.join(sys.argv[3:]) + '.eps', format='eps', ppi=300)
