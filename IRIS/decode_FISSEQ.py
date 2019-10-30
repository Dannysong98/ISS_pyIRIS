#!/usr/bin/env python3
"""
This model is used to decode the result of Ke into FISSEQ.
"""

from numpy import (array,
                   uint8)


class graph:
    P_map = array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                   [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                   [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                   [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]], dtype=uint8)

    M_lab_coor = {'AA': 0, 'AT': 1, 'AC': 2, 'AG': 3,
                  'TA': 4, 'TT': 5, 'TC': 6, 'TG': 7,
                  'CA': 8, 'CT': 9, 'CC': 10, 'CG': 11,
                  'GA': 12, 'GT': 13, 'GC': 14, 'GG': 15}

    code = {'A': ['AA', 'TT', 'CC', 'GG'],
            'T': ['AT', 'TA', 'CG', 'GC'],
            'C': ['AC', 'CA', 'TG', 'GT'],
            'G': ['AG', 'GA', 'TC', 'CT']}

    def __init__(self):
        """"""
        self.p_node = ''
        self.n_node = ''

        self.score = 0
        self.decoded_seq = ''

    def decode_trace(self, ext, seq):
        """"""
        self.decoded_seq = seq

        for sig in ext:
            for pair in graph.code[sig]:
                self.score = graph.P_map[graph.M_lab_coor[self.decoded_seq[-2:]], graph.M_lab_coor[pair]]

                if self.score == 1:
                    self.decoded_seq += pair[-1]
                    break


if __name__ == '__main__':
    pass
