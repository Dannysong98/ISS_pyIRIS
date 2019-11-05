#!/usr/bin/env python3


from sys import argv
from os.path import basename

from IRIS.decode_FISSEQ import assembly_graph


def parse_FISSEQ(file_path):
    """"""
    out_prefix = str(basename(file_path)).split('.')[0]

    with open(out_prefix + '.FISSEQ.txt', 'wt') as OFH:
        g = assembly_graph()

        with open(file_path, 'rt') as IFH:
            for ln in IFH:
                ln = ln .split()
                if 'N' in ln[1]:
                    continue

                for p_base in ('AA', 'TT', 'CC', 'GG'):
                    g.decode_trace(p_base, ln[1])
                    qul = 'II' + ln[2]

                    print('%s_%s\t%s\t%s\t%s\t%s' % (ln[0], p_base, g.decoded_seq, qul, ln[3], ln[4]), file=OFH)


if __name__ == '__main__':
    parse_FISSEQ(argv[1])
