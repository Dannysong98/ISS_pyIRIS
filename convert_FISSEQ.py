#!/usr/bin/env python3


from sys import argv
from os.path import basename

from IRIS.decode_FISSEQ import assembly_graph


def parse_FISSEQ(file_path, start_seq):
    """"""
    out_prefix = str(basename(file_path)).split('.')[0]

    with open(out_prefix + '.FISSEQ.txt', 'wt') as OFH:
        g = assembly_graph()

        with open(file_path, 'rt') as IFH:
            for ln in IFH:
                ln = ln .split()
                if 'N' in ln[1]:
                    continue

                g.decode_trace(start_seq, ln[1])

                print('%s\t%s\t%s\t%s\t%s' % (ln[0], g.decoded_seq, 'II' + ln[2], ln[3], ln[4]), file=OFH)


if __name__ == '__main__':
    parse_FISSEQ(argv[1], argv[2])
