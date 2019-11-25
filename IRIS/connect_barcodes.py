#!/usr/bin/env python3
"""
This module is used to connect bases from different cycles in the same location to form barcode sequence.

In this model, the connection is made as a class, the 'BarcodeCube', of which, three method will be employed. The
method 'collect_called_bases' is used to record the called bases in each cycle, and 'filter_blobs_list' is used to
filter the bad or indistinguishable blobs by mapping them into a mask layer. At last, the method 'calling_adjust' is
used to connect the bases as barcodes, by anchoring the coordinates of blobs in reference layer, and search their 20x20
region in each cycle.
"""


from sys import stderr
from numpy import sqrt


class BarcodeCube:
    def __init__(self):
        """
        This method will initialize three members, '__all_blobs_list' store all blobs' id, 'bases_cube' is
        a list stores the dictionary of bases in each cycle, and 'adjusted_bases_cube' is a list stores
        the dictionary of bases in each cycle, with error rate adjusted.
        """
        self.__all_blobs_list = []

        self.bases_cube = []
        self.adjusted_bases_cube = []

    def collect_called_bases(self, called_base_in_one_cycle):
        """
        This method is used to record the called bases in each cycle.

        A list which store all blobs' id and a list which store the dictionary of bases in each cycle.

        :param called_base_in_one_cycle: The dictionary of bases in a cycle.
        :return: NONE
        """
        self.__all_blobs_list = [_ for _ in called_base_in_one_cycle.keys() if 'N' not in called_base_in_one_cycle[_]]
        self.bases_cube.append(called_base_in_one_cycle)

    ###############################################
    # Old redundancy-filtering strategy (ABANDON) #
    ###############################################
    # def filter_blobs_list(self, f_background):
    #     """
    #     This method is used to filter the recorded bases in the called base list.
    #
    #     A new list will be generated, which store the filtered id of bases
    #
    #     :param f_background: The background image for ensuring the shape of mask layer.
    #     :return: NONE
    #     """
    #     blobs_mask = zeros(f_background.shape, dtype=uint8)
    #
    #     new_coor = set()
    #
    #     for coor in self.__all_blobs_list:
    #         r = int(coor[1:6].lstrip('0'))
    #         c = int(coor[7:].lstrip('0'))
    #
    #         blobs_mask[r:(r + 2), c:(c + 2)] = 255
    #
    #     _, contours, _ = findContours(blobs_mask, RETR_LIST, CHAIN_APPROX_NONE)
    #
    #     for cnt in contours:
    #         M = moments(cnt)
    #
    #         if M['m00'] != 0:
    #             cr = abs(int(M['m01'] / M['m00']))
    #             cc = abs(int(M['m10'] / M['m00']))
    #
    #             new_coor.add(str('r' + ('%05d' % cr) + 'c' + ('%05d' % cc)))
    #
    #     self.__all_blobs_list = new_coor
    ###############################################

    def filter_blobs_list(self):
        """
        This method is used to filter the recorded bases in the called base list.

        A new list will be generated, which store the filtered id of bases
        """
        new_coor = self.__all_blobs_list

        for coor in self.__all_blobs_list:
            r = int(coor[1:6].lstrip('0'))
            c = int(coor[7:].lstrip('0'))

            N = 3  # 8x8
            for row in range(r - N, r + (N + 2)):
                for col in range(c - N, c + (N + 2)):
                    if row == r and col == c:
                        continue

                    if 'r%05dc%05d' % (row, col) in self.__all_blobs_list:
                        new_coor.remove('r%05dc%05d' % (row, col))

        self.__all_blobs_list = set(new_coor)

    def calling_adjust(self):
        """
        This method is used to connect bases into barcodes, by anchoring the coordinates of blobs in reference layer,
        and search their 20x20 region in each cycle.

        :return: NONE
        """
        def __check_greyscale(all_blobs_list, bases_cube, adjusted_bases_cube, cycle_serial):
            """"""
            adjusted_bases_cube[cycle_serial] = {}

            for ref_coordinate in all_blobs_list:
                r = int(ref_coordinate[1:6].lstrip('0'))
                c = int(ref_coordinate[7:].lstrip('0'))

                max_qul_base = 'N'
                min_err_rate = float(1)

                ###################################################################################################
                # It will search a 20x20 region to connect bases from each cycle in ref-coordinates               #
                #                                                                                                 #
                # Process of registration almost align all location of cycles the same, but at pixel level, this  #
                # registration is not accurate enough. Here, we choose a simple approach to solve this problem,   #
                # we get locations of blobs from a reference image layer, then to search a NxN (20x20 by default) #
                # region in those cycles that need to be connected. This approach should not only solve this      #
                # problem but also bring few false positive in output                                             #
                ###################################################################################################
                N = 3  # 8x8
                ########
                # N = 1  # Alternative option, 4x4
                # N = 2  # Alternative option, 6x6
                # N = 4  # Alternative option, 10x10
                # N = 5  # Alternative option, 12x12
                # N = 9  # Alternative option, 20x20
                ########
                # N = n  # Alternative option, ((n + 1) * 2)x((n + 1) * 2)

                for row in range(r - N, r + (N + 2)):
                    for col in range(c - N, c + (N + 2)):
                        coor = 'r%05dc%05d' % (row, col)

                        if coor in bases_cube[cycle_serial]:
                            ######################################################################
                            # Adjust of error rate of each coordinate by the Pythagorean theorem #
                            # This function can be off if no need                                #
                            ######################################################################
                            error_rate = bases_cube[cycle_serial][coor][1]
                            D = sqrt((row - r) ** 2 + (col - c) ** 2)

                            adj_err_rate = sqrt(((error_rate * D) ** 2) + (error_rate ** 2))
                            ########
                            # adj_err_rate = error_rate  # Alternative option
                            ######################################################################

                            if adj_err_rate > 1:
                                adj_err_rate = float(1)

                            if adj_err_rate < min_err_rate:
                                max_qul_base = bases_cube[cycle_serial][coor][0]
                                min_err_rate = adj_err_rate
                ###################################################################################################

                adjusted_bases_cube[cycle_serial].update({ref_coordinate: [max_qul_base, min_err_rate]})

        if len(self.bases_cube) > 0:
            for cycle_id in range(0, len(self.bases_cube)):
                self.adjusted_bases_cube.append({})

                __check_greyscale(self.__all_blobs_list, self.bases_cube, self.adjusted_bases_cube, cycle_id)

            if len(self.bases_cube) == 1:
                print('There is only one cycle in this run', file=stderr)


if __name__ == '__main__':
    pass
