import os
import pdb
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn import preprocessing


def preprocess_data(x_data, method):
    # preprocess of traces
    if method == 'norm':     # 'horizontal_standardization':
        print('[LOG] -- using {} method to preprocessing the data.'.format(method))
        mn = np.repeat(np.mean(x_data, axis=1, keepdims=True), x_data.shape[1], axis=1)
        std = np.repeat(np.std(x_data, axis=1, keepdims=True), x_data.shape[1], axis=1)
        x_data = (x_data - mn)/std
    elif method == 'scaling':    #  'horizontal_scaling':
        print('[LOG] -- using {} method to preprocessing the data.'.format(method))
        scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(x_data.T)
        x_data = scaler.transform(x_data.T).T
    else:
        print('[LOG] -- not perform preprocessing method to the data.')

    return x_data


sbox = [
    # 0    1    2    3    4    5    6    7    8    9    a    b    c    d    e    f
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,  # 0
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,  # 1
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,  # 2
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,  # 3
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,  # 4
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,  # 5
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,  # 6
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,  # 7
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,  # 8
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,  # 9
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,  # a
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,  # b
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,  # c
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,  # d
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,  # e
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16  # f
]

MSB_byte=[0,0,0,0,1,0,0,1,0,0,0,0,1,1,1,0,1,1,1,0,1,0,0,
 1,1,1,1,1,1,1,0,1,1,1,1,0,0,0,1,1,0,1,1,1,0,1,
 0,0,0,1,0,1,0,1,0,1,0,0,1,1,1,0,1,0,0,1,0,0,0,
 0,0,1,0,0,1,1,0,1,0,1,0,1,0,1,0,1,1,0,0,1,1,0,
 0,0,0,1,1,1,1,1,0,0,0,1,0,1,0,0,0,0,1,1,0,1,0,
 1,1,1,0,1,1,1,1,0,0,1,1,1,1,0,0,1,0,1,0,0,1,1,
 0,0,0,0,0,0,0,1,0,1,0,0,1,1,0,1,1,0,1,0,0,1,1,
 0,0,0,0,0,0,0,1,1,1,0,1,1,1,0,1,1,0,0,1,1,0,1,
 0,0,1,1,0,0,1,0,1,0,0,0,0,1,1,1,1,1,0,0,0,1,1,
 1,0,0,1,0,0,0,1,0,0,0,0,1,1,1,0,1,1,1,1,0,0,1,
 1,1,1,0,1,1,1,0,0,1,1,1,1,0,1,1,0,0,0,1,0,0,1,
 0,1,0]
def calc_hamming_weight(n):
    return bin(n).count("1")


def get_HW():
    HW = []
    for i in range(0, 256):
        hw_val = calc_hamming_weight(i)
        HW.append(hw_val)
    return HW


def aes_internal(inp_data_byte, key_byte):
    inp_data_byte = int(inp_data_byte)
    return sbox[inp_data_byte ^ key_byte]


def create_hw_label_mapping():
    ''' this function return a mapping that maps hw label to number per class '''
    HW = defaultdict(list)
    for i in range(0, 256):
        hw_val = calc_hamming_weight(i)
        HW[hw_val].append(i)
    return HW


def get_labels(plain_text, key_byte, target_byte, leakage_model):
    if 'HW' == leakage_model:
        HW = get_HW()

    labels = []
    for i in range(plain_text.shape[0]):
        text_i = plain_text[i]
        label = aes_internal(text_i[target_byte], key_byte)
        if 'HW' == leakage_model:
            label = HW[label]
        elif 'MSB' == leakage_model:
            label = MSB_byte[label]
        labels.append(label)

    if 'HW' == leakage_model:
        try:
            assert(set(labels) == set(list(range(9))))
        except Exception:
            print('[LOG] -- not all class have data: ', set(labels))
    elif 'MSB' == leakage_model:
        try:
            assert(set(labels) == set([0,1]))
        except Exception:
            print('[LOG] -- not all class have data: ', set(labels))
    else:
        try:
            assert(set(labels) == set(range(256)))
        except Exception:
            print('[LOG] -- not all class have data: ', set(labels))
    labels = np.array(labels)
    return labels

def shift_the_data(shifted, attack_window, trace_mat, textin_mat):
    start_idx, end_idx = attack_window[0], attack_window[1]

    if shifted:
        print('[LOG] -- data will be shifted in range: ', [0, shifted])
        shifted_traces = []
        for i in range(textin_mat.shape[0]):
            random_int = random.randint(0, shifted)
            trace_i = trace_mat[i, start_idx+random_int:end_idx+random_int]
            shifted_traces.append(trace_i)
        trace_mat = np.array(shifted_traces)
    else:
        print('[LOG] -- no random delay apply to the data')
        trace_mat = trace_mat[:, start_idx:end_idx]

    return trace_mat, textin_mat

def unpack_data(whole_pack):
    try:
        traces, plain_text, key = whole_pack['power_trace'], whole_pack['plain_text'], whole_pack['key']
    except KeyError:
        try:
            traces, plain_text, key = whole_pack['power_trace'], whole_pack['plaintext'], whole_pack['key']
        except KeyError:
            traces, plain_text, key = whole_pack['trace_mat'], whole_pack['textin_mat'], whole_pack['key']
    return traces, plain_text, key


def load_data_base(whole_pack, attack_window, method, trace_num=0, shifted=0):
    if isinstance(attack_window, str):
        tmp = attack_window.split('_')
        attack_window = [int(tmp[0]), int(tmp[1])]

    traces, plain_text, key = unpack_data(whole_pack)

    if trace_num:
        traces = traces[:trace_num, :]
        plain_text = plain_text[:trace_num, :]

    traces, plain_text = shift_the_data(shifted, attack_window, traces, plain_text)

    if method:
        traces = preprocess_data(traces, method)
    return traces, plain_text, key
def load_data_base_test(whole_pack, attack_window, method, trace_num=0, shifted=0):
    if isinstance(attack_window, str):
        tmp = attack_window.split('_')
        attack_window = [int(tmp[0]), int(tmp[1])]

    traces, plain_text, key = unpack_data(whole_pack)

    if trace_num:
        traces = traces[-trace_num:, :]
        plain_text = plain_text[-trace_num:, :]

    traces, plain_text = shift_the_data(shifted, attack_window, traces, plain_text)

    if method:
        traces = preprocess_data(traces, method)
    return traces, plain_text, key
def create_df(power_traces, labels, n):
    """
    This function creates a dataframe from the numpy array and generates the subset of the dataset which is used for
    training the feature extractor
    :param power_traces: The power traces used for training the model
    :param power_traces_labels: The labels corresponding to the power traces
    :param n: Number of traces to be selected for each class
    :return: the subset of the dataset
    """
    # creating dataframe of the data for selecting subset of the dataset
    y_df = pd.DataFrame(data=labels, columns=['labels'])
    x_df = pd.DataFrame(data=power_traces)
    frames = [y_df, x_df]
    # creating the dataframe
    all_data = pd.concat(frames, axis=1)
    all_data = all_data.sort_values(by=['labels'])

    # select N samples from each class
    grouped = all_data.groupby(['labels'], as_index=False)
    all_data = grouped.apply(lambda frame: frame.sample(n))
    all_data.index = all_data.index.droplevel(0)
    all_data = all_data.reset_index(drop=True)

    # separating power traces and labels
    power_traces = all_data.iloc[:, 1:]
    power_traces = power_traces.to_numpy()
    labels = all_data['labels']
    print('shape of the power traces to be used for training: ', power_traces.shape)
    return [power_traces, labels, all_data]


def data_info(power_traces_shape, plain_text_shape, key):
    print('shape of the plain text matrix : ', plain_text_shape)
    print('shape of the power trace matrix: ', power_traces_shape)
    print('Encryption key: ', key)
    print('-' * 90)


def sanity_check(input_layer_shape, X_profiling):
    if input_layer_shape[1] != X_profiling.shape[1]:
        print("Error: model input shape %d instead of %d is not expected ..." % (input_layer_shape[1], len(X_profiling[0])))
        sys.exit(-1)
    # Adapt the data shape according our model input
    if len(input_layer_shape) == 2:
        # This is a MLP
        Reshaped_X_profiling = X_profiling
    elif len(input_layer_shape) == 3:
        # This is a CNN: expand the dimensions
        Reshaped_X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1))
    else:
        print("Error: model input shape length %d is not expected ..." % len(input_layer_shape))
        sys.exit(-1)
    return Reshaped_X_profiling


def binary_search(arr):
    # we need to do it in a binary search style
    low = 0
    high = arr.shape[0]
    mid = 0

    if 0 == sum(arr):
        return 0
    rtn = high
    while low <= high:
        mid = (low + high) // 2
        # if later part summation is equal to 0
        if 0 == sum(arr[mid:]):
            high = mid - 1
            rtn = mid
        # if first part summation is equal to 0
        else:
            low = mid + 1

    return rtn


def compute_min_rank(ranking_list):
    ''' try to find the last value that not convergence to 0 '''
    ranking_list = np.array(ranking_list)
    num = binary_search(ranking_list)
    return num
