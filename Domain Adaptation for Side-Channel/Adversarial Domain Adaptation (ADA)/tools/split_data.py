import os
import sys
import numpy as np
import argparse


def loadData(opts):
    fpath = opts.input
    print('load file from file: ', fpath)
    whole_pack = np.load(fpath)
    try:
        trace_array, textin_array, key = whole_pack['power_trace'], whole_pack['plain_text'], whole_pack['key']
    except Exception:
        try:
            trace_array, textin_array, key = whole_pack['trace_mat'], whole_pack['textin_mat'], whole_pack['key']
        except Exception:
            trace_array, textin_array, key = whole_pack['power_trace'], whole_pack['plaintext'], whole_pack['key']

    return trace_array, textin_array, key


def main(opts):
    trace_array, textin_mat, key = loadData(opts)

    trace_array1 = trace_array[-opts.trace_num:, :]
    textin_mat1 = textin_mat[-opts.trace_num:, :]

    droot = os.path.dirname(opts.input)
    fpath1 = os.path.join(droot, 'val_diff_key.npz')

    np.savez(fpath1, trace_mat=trace_array1, textin_mat=textin_mat1, key=key)

    print('save data to path: {}'.format(fpath1))


def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='')
    parser.add_argument('-tn', '--trace_num', default=10000, type=int, help='')
    opts = parser.parse_args()
    return opts


if __name__ == '__main__':
    opts = parseArgs(sys.argv)
    main(opts)
