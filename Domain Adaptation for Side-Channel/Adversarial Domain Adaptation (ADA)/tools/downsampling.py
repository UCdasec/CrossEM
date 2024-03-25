import os
import sys
import argparse
import pdb

import numpy as np


def load_data(opts):
    whole_pack = np.load(opts.input)
    trace_mat, textin_mat = whole_pack['power_trace'], whole_pack['plain_text']
    key = whole_pack['key']

    attack_window_str = opts.attack_window
    tmp = attack_window_str.split('_')
    attack_window = [int(tmp[0]), int(tmp[1])]

    trace_mat = trace_mat[:, attack_window[0]:attack_window[1]]
    return trace_mat, textin_mat, key, attack_window


def pooling(pool_style):
    # calculate the parameters for the pooling
    pass
    # perform the pooling operation
    if 'max' == pool_style:
        layers = [
            MaxPooling1D(pool_size, strides=1, padding='valid')
        ]
    if 'ave' == pool_style:
        layers = [
            AveragePooling1D(pool_size, strides=1, padding='valid')
        ]
    model = Sequential(layers)



def downsampling(datas, outdim):
    height, width = datas.shape[0], datas.shape[1]
    blocks = width // outdim
    mid_idx = blocks // 2
    selected_cols = list(range(mid_idx, width, blocks))
    selected_cols = selected_cols[:outdim]
    new_data = datas[:, selected_cols]
    assert(new_data.shape[1] == outdim)
    return new_data


def copy_weight(feat_model, full_model):
    pass


def sae(model_path, datas, outdim, params):
    # load model
    inp_shape = (datas.shape[1],)
    inp = Input(inp_shape)
    feat_model = Model(inp, sae.encoder(inp, params))
    full_model = load_model(model_path)
    feat_model = copy_weight(feat_model, full_model)

    predicts = model.predict(datas)
    return predicts


def main(opts):
    print('[LOG] -- loading the data for file: ', opts.input)
    trace_mat, textin_mat, key, attack_window = load_data(opts)
    if 'pooling' == opts.method:
        new_trace_mat = pca(trace_mat, textin_mat, key)
    elif 'sae' == opts.method:
        new_trace_mat = sae()
    elif 'downsampling' == opts.method:
        new_trace_mat = downsampling(trace_mat, opts.outdim)
    else:
        raise ValueError()

    oldname = os.path.basename(opts.input).split('.')[0]
    spath = os.path.join(opts.output, '{}_reduced_{}.npz'.format(oldname, opts.outdim))
    attack_window = [0, new_trace_mat.shape[1]]
    np.savez(spath, trace_mat=new_trace_mat, textin_mat=textin_mat, key=key, attack_window=attack_window)
    print('[LOG] -- the reduced data is save to fpath {}'.format(spath))


def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='')
    parser.add_argument('-o', '--output', help='specify the folder to store the output')
    parser.add_argument('-m', '--method', choices={'pooling', 'sae', 'downsampling'}, help='')
    parser.add_argument('-od', '--outdim', type=int, default=1200, help='')
    parser.add_argument('-aw', '--attack_window', default='', help='')
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parseArgs(sys.argv)
    main(opts)
