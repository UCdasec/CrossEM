import numpy as np
import argparse
import os
import sys


def main(opts):
    whole_pack = np.load(opts.input)
    key = whole_pack['key']
    hex_key = [hex(val)[2:] for val in key]
    hex_key_str = '0x' + ', '.join(hex_key)
    print(hex_key_str)

def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='')
    opts = parser.parse_args()
    return opts


if __name__ == '__main__':
    opts = parseArgs(sys.argv)
    main(opts)
