import os
import sys
import argparse
import pdb

import numpy as np
import tensorflow as tf


def loadData(opts):
    whole_pack = np.load(opts.input)
    trace_mat = whole_pack['trace_mat']
    textin_mat = whole_pack['textin_mat']
    key = whole_pack['key']
    attack_window = whole_pack['attack_window']

    sIdx, eIdx = attack_window[0], attack_window[1]
    trace_mat = trace_mat[:, sIdx:eIdx]

    return trace_mat, textin_mat, key


def pooling(inp_map, pool_size, stride=1, mode='max'):
    '''
    Input:
        inp_map - input array of the pooling layer
        pool_size - X-size (equivalent to Y-size) of receptive field
        stride - the stride size between successive pooling squares
    Output:
        output - output array of the pooling layer
    Padding mode - 'edge'
    '''
    


def sampling(datas, out_dim):
    ''''''
    pass


def ave_pooling(datas, out_dim):
    ''' calculate the avg pooling by using the tensorflow '''
    inp_dim = datas.shape[1]

    avg_pool_model = tf.Sequential()
    avg_pool_model.add(tf.keras.layers.AveragePooling1D())

    avg_pool_model.pre


def max_pooling():
    pass


def pac():
    pass


def ae():
    pass


def main(opts):
    # load the data
    trace_mat, textin_mat, key, attack_window = loadData(opts)

    # choose methods
    if 'sampling':
        reduction_func = sampling()
    elif 'ave_pool':
        reduction_func = ave_pooling()
    elif 'max_pool':
        reduction_func = max_pooling()
    elif 'pca':
        reduction_func = pac()
    elif 'ae':
        reduction_func = ae()
    else:
        raise ValueError()


    # perform dimension reduction
    new_data = reduction_func()

    # save the data
    attack_window = [0, 1200]
    np.savez(spath, tract_mat=new_data, textin_mat=textin_mat, key=key, attack_window=attack_window)

    print('[LOG] -- all done!')


def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='')
    parser.add_argument('-o', '--output', help='')
    parser.add_argument('-m', '--method', help='',
                        choices={'sampling', 'ave_pool', 'max_pool', 'pca', 'ae'})

    opts = parser.parse_args()
    return opts


if __name__ == '__main__':
    opts = parseArgs(sys.argv)
    main(opts)

