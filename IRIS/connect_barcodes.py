#!/usr/bin/env python3
"""
This model is used to connect the base from different cycles.

In this model, the connection is made as a class, the 'BarcodeCube', of which, three method will be employed. The method
'collect_called_bases' is used to record the called bases in each cycle, and 'filter_blobs_list' is used to filter the
bad or indistinguishable blobs by mapping them into a mask layer. At last, the method 'calling_adjust' is used to
connect the bases as barcodes, by anchoring the coordinates of blobs in cycle 1, and search their 12x12 region in other
cycles, respectively.
"""


from sys import stderr
from cv2.cv2 import (findContours, moments,
                     RETR_LIST,
                     CHAIN_APPROX_NONE)
from numpy import (zeros,
                   sqrt,
                   uint8)


class BarcodeCube:
    def __init__(self):
        """
        This method will initialize three members, the '__all_blobs_list' store all the id of blobs, the 'bases_cube' is
        A list which store the dictionary of bases in each cycle, while the 'adjusted_bases_cube' is A list which store
        the dictionary of bases in each cycle, with error rate adjusted.
        """
        self.__all_blobs_list = set()

        self.bases_cube = []
        self.adjusted_bases_cube = []

    def collect_called_bases(self, f_called_base_in_one_cycle):
        """
        This method is used to record the called bases in each cycle.

        A list which store all the id of bases and A list which store the dictionary of bases in each cycle.

        :param f_called_base_in_one_cycle: The dictionary of bases in a cycle.
        :return: NONE
        """
        self.__all_blobs_list.update([_ for _ in f_called_base_in_one_cycle.keys()
                                      if 'N' not in f_called_base_in_one_cycle[_]])
        self.bases_cube.append(f_called_base_in_one_cycle)

    def filter_blobs_list(self, f_background):
        """
        This method is used to filter the recorded bases in the called base list.

        A new list will be generated, which store the filtered id of bases

        :param f_background: The background image for ensuring the shape of mask layer.
        :return: NONE
        """
        blobs_mask = zeros(f_background.shape, dtype=uint8)

        new_coor = set()

        for coor in self.__all_blobs_list:
            r = int(coor[1:6].lstrip('0'))
            c = int(coor[7:].lstrip('0'))

            blobs_mask[r:(r + 2), c:(c + 2)] = 255

            _, contours, _ = findContours(blobs_mask, RETR_LIST, CHAIN_APPROX_NONE)

            for cnt in contours:
                M = moments(cnt)

                if M['m00'] != 0:
                    cr = abs(int(M['m01'] / M['m00']))
                    cc = abs(int(M['m10'] / M['m00']))

                    new_coor.add(str('r' + ('%05d' % cr) + 'c' + ('%05d' % cc)))

        self.__all_blobs_list = new_coor

    def calling_adjust(self):
        """
        This method is used to connect the bases as barcodes, by anchoring the coordinates of blobs in cycle 1, and
        search their 12x12 region in other cycles, respectively.

        :return: NONE
        """
        def __check_greyscale(f_all_blobs_list, f_bases_cube, f_adjusted_bases_cube, f_cycle_id):
            """"""
            f_adjusted_bases_cube[f_cycle_id] = {}

            for ref_coordinate in f_all_blobs_list:

                r = int(ref_coordinate[1:6].lstrip('0'))
                c = int(ref_coordinate[7:].lstrip('0'))

                max_qul_base = 'N'
                min_err_rate = float(1)

                for row in range(r - 5, r + 7):
                    for col in range(c - 5, c + 7):
                        coor = str('r' + ('%05d' % row) + 'c' + ('%05d' % col))

                        if coor in f_bases_cube[f_cycle_id]:
                            # Adjusting of Error Rate #
                            error_rate = f_bases_cube[f_cycle_id][coor][1]
                            D = sqrt((row - r) ** 2 + (col - c) ** 2)
                            adj_err_rate = sqrt(((error_rate * D) ** 2) + (error_rate ** 2))

                            if adj_err_rate > 1:
                                adj_err_rate = float(1)

                            if adj_err_rate < min_err_rate:
                                max_qul_base = f_bases_cube[f_cycle_id][coor][0]
                                min_err_rate = adj_err_rate

                f_adjusted_bases_cube[f_cycle_id].update({ref_coordinate: [max_qul_base, min_err_rate]})

        if len(self.bases_cube) > 0:
            for cycle_id in range(0, len(self.bases_cube)):
                self.adjusted_bases_cube.append({})

                __check_greyscale(self.__all_blobs_list, self.bases_cube, self.adjusted_bases_cube, cycle_id)

            if len(self.bases_cube) == 1:
                print('There is only one cycle in this run', file=stderr)


if __name__ == '__main__':
    pass
