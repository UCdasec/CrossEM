import os
import numpy as np
import warnings
import random
import pdb
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
warnings.filterwarnings("ignore")

#import tools.key_rank_hw as key_rank_hw
#import tools.key_rank as key_rank


sbox = [
    # 0    1    2    3    4    5    6    7    8    9    a    b    c    d    e    f 
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76, # 0
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0, # 1
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15, # 2
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75, # 3
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84, # 4
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf, # 5
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8, # 6
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2, # 7
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73, # 8
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb, # 9
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79, # a
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08, # b
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a, # c
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e, # d
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf, # e
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16  # f
]


def aes_internal(inp_data_byte, key_byte):
    inp_data_byte = int(inp_data_byte)
    key_byte = int(key_byte)
    return sbox[inp_data_byte ^ key_byte]


def calc_hamming_weight(n):
    return bin(n).count("1")


def get_HW():
    HW = []
    for i in range(0, 256):
        hw_val = calc_hamming_weight(i)
        HW.append(hw_val)
    return HW


def create_hw_label_mapping():
    ''' this function return a mapping that maps hw label to number per class '''
    HW = defaultdict(list)
    for i in range(0, 256):
        hw_val = calc_hamming_weight(i)
        HW[hw_val].append(i)
    return HW


def shift_the_data(shifted, attack_window, trace_mat, textin_mat, trace_num=0):
    start_idx, end_idx = attack_window[0], attack_window[1]
    if trace_num:
        trace_mat = trace_mat[:trace_num, :]
        textin_mat = textin_mat[:trace_num, :]

    if shifted:
        print('[LOG] -- data will be shifted in range: ', [0, shifted])
        shifted_traces, new_textins = [], []
        for i in range(textin_mat.shape[0]):
            textin_i = textin_mat[i, :]

            random_int = random.randint(0, shifted)
            trace_i = trace_mat[i, start_idx+random_int:end_idx+random_int]

            shifted_traces.append(trace_i)
            new_textins.append(textin_i)
        shifted_traces, new_textins = np.array(shifted_traces), np.array(new_textins)
        return shifted_traces, new_textins
    else:
        trace_mat = trace_mat[:, start_idx:end_idx]
    return trace_mat, textin_mat


def load_base(whole_pack):
    key = whole_pack['key']
    try:
        trace_mat, textin_mat = whole_pack['trace_mat'], whole_pack['textin_mat']
    except Exception:
        try:
            trace_mat, textin_mat = whole_pack['power_trace'], whole_pack['plain_text']
        except Exception:
            try:
                trace_mat, textin_mat = whole_pack['power_trace'], whole_pack['plaintext']
            except Exception:
                trace_mat, textin_mat = whole_pack['power_trace'], whole_pack['textin_array']

    return trace_mat, textin_mat, key


class PowertraceDataset(Dataset):
    ''' Power trace dataset. '''
    def __init__(self, whole_pack, target_byte, attack_window, leakage_model, shifted=0, trace_num=0):

        self.target_byte = target_byte
        self.shifted = shifted
        self.leakage_model = leakage_model
        self.HW = get_HW()

        tmp = attack_window.split('_')
        self.attack_window = [int(tmp[0]), int(tmp[1])]
        print('[LOG] -- using self-defined attack window, attack window is: ', self.attack_window)

        ori_trace_mat, ori_textin_mat, key = load_base(whole_pack)
        self.key = key
        self.key_byte = key[target_byte]
        trace_mat, self.textin_mat = shift_the_data(shifted, self.attack_window, ori_trace_mat, ori_textin_mat, trace_num)

        start_idx, end_idx = self.attack_window[0], self.attack_window[1]
        self.inp_shape = (1, end_idx-start_idx)

        self.trace_mat = trace_mat[:, np.newaxis, :]

    def __len__(self):
        return self.trace_mat.shape[0]

    def __getitem__(self, idx):
        trace = self.trace_mat[idx, :]
        textin = self.textin_mat[idx, :]
        label = aes_internal(textin[self.target_byte], self.key_byte)
        if 'HW' == self.leakage_model:
            label = self.HW[label]
        return trace, label, textin

    def get_inp_shape(self):
        return self.inp_shape

    def get_data_key(self):
        return self.key


def loader(dpath, target_byte, attack_window, batch_size, trace_num, shifted, leakage_model, kwargs):
    ''' load source domain data '''
    print('[LOG] -- load the source domain data from path {} now...'.format(dpath))
    whole_pack = np.load(dpath)
    src_data = PowertraceDataset(whole_pack, target_byte, attack_window, leakage_model, shifted=shifted, trace_num=trace_num)
    train_loader = DataLoader(src_data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    inp_shape = src_data.get_inp_shape()
    return train_loader, inp_shape


def test_loader(dpath, target_byte, attack_window, batch_size, trace_num, shifted, leakage_model, kwargs):
    ''' load source domain data '''
    print('[LOG] -- load the source domain data now...')
    whole_pack = np.load(dpath)
    trace_mat, textin_mat, key = load_base(whole_pack)

    test_trace_mat = trace_mat[:20000, :]
    test_textin_mat = textin_mat[:20000, :]

    test_pack = {'trace_mat': test_trace_mat,
                 'textin_mat': test_textin_mat,
                 'key': key}

    src_data = PowertraceDataset(test_pack, target_byte, attack_window, leakage_model, shifted=shifted, trace_num=trace_num)
    train_loader = DataLoader(src_data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    inp_shape = src_data.get_inp_shape()
    return train_loader, inp_shape


def load_inp(dpath, target_byte, attack_window, batch_size, trace_num, shifted, leakage_model, kwargs):
    ''' load data from straight forward cnn model'''
    print('[LOG] -- loading the train - val data for the cnn method...')
    whole_pack = np.load(dpath)
    trace_mat, textin_mat, key = load_base(whole_pack)
    if trace_num:
        trace_mat = trace_mat[:trace_num, :]
        textin_mat = textin_mat[:trace_num, :]

    val_ratio = 0.2
    idx1 = int(trace_mat.shape[0] * (1 - val_ratio))
    idx2 = trace_mat.shape[0] - idx1

    train_trace_mat = trace_mat[:idx1, :]
    train_textin_mat = textin_mat[:idx1, :]

    test_trace_mat = trace_mat[idx1:idx1+idx2, :]
    test_textin_mat = textin_mat[idx1:idx1+idx2, :]

    train_pack = {'trace_mat': train_trace_mat,
                  'textin_mat': train_textin_mat,
                  'key': key}
    test_pack = {'trace_mat': test_trace_mat,
                 'textin_mat': test_textin_mat,
                 'key': key}

    train_data = PowertraceDataset(train_pack, target_byte, attack_window, leakage_model, shifted=shifted, trace_num=trace_num)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)

    test_data = PowertraceDataset(test_pack, target_byte, attack_window, leakage_model, shifted=shifted, trace_num=trace_num)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)

    inp_shape = train_data.get_inp_shape()
    return train_loader, test_loader, inp_shape


# ###############################################
# ############ plot rank curve figure ###########
# ###############################################
def plot_figure(x, y, dataset_name, fig_save_name):
    plt.title('ranking curve of dataset: ' + dataset_name)
    plt.xlabel('number of traces')
    plt.ylabel('rank')
    plt.grid(True)
    plt.plot(x, y)
    plt.savefig(fig_save_name)
    plt.show(block=False)
    plt.figure()


def shuffle_and_pick(mat1, mat2, choice_num):
    perm = np.random.permutation(mat1.shape[0])
    new_mat1 = mat1[perm]
    new_mat2 = mat2[perm]

    new_mat1 = new_mat1[:choice_num]
    new_mat2 = new_mat2[:choice_num]
    return new_mat1, new_mat2


def ranking_curve(opts, proba_mat, textin_mat, key, target_byte, rank_dir):
    min_trace_idx = 0
    max_trace_idx = 3000
    rank_step = opts.rank_step

    # run rank key curve function
    assert(proba_mat.shape[0] == textin_mat.shape[0])
    y_list = []
    for i in tqdm(range(100)):
        proba_mat_x, textin_mat_x = shuffle_and_pick(proba_mat, textin_mat, choice_num=max_trace_idx)
        if 'HW' == opts.leakage_model:
            f_ranks = key_rank_hw.full_ranks(proba_mat_x, key, textin_mat_x, min_trace_idx, max_trace_idx, target_byte, rank_step)
        else:
            f_ranks = key_rank.full_ranks(proba_mat_x, key, textin_mat_x, min_trace_idx, max_trace_idx, target_byte, rank_step)

        # We plot the results  f_ranks[i] = [t, real_key_rank]
        x = [f_ranks[i][0] for i in range(0, f_ranks.shape[0])]
        y_tmp = [f_ranks[i][1] for i in range(0, f_ranks.shape[0])]
        y_list.append(y_tmp)

    y_arr = np.array(y_list)
    y = np.mean(y_arr, axis=0)

    fig_save_name = os.path.join(rank_dir, 'rank_byte_{}.png'.format(target_byte))
    dataset_name = os.path.basename(opts.input)
    plot_figure(x, y, dataset_name, fig_save_name)
    print('[LOG] -- figure save to file: {}'.format(fig_save_name))

    # save the raw data
    outfile = os.path.join(rank_dir, 'rank_byte_{}.npz'.format(target_byte))
    np.savez(outfile, x=x, y=y)
    print('[LOG] -- raw data save to path: {}'.format(outfile))

