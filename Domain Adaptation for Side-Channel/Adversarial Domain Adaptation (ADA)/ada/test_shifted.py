import os
import sys
import pdb
import argparse
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable

import data_loader
import models
import tools.process_data as process_data
import mytools.draw_base as draw_base


# ###############################################
# ############ plot rank curve figure ###########
# ###############################################
def plot_figure(x, y, model_file_name, dataset_name, fig_save_name):
    plt.title('Performance of ' + model_file_name + ' against ' + dataset_name)
    plt.xlabel('number of traces')
    plt.ylabel('rank')
    plt.grid(True)
    plt.plot(x, y)
    plt.savefig(fig_save_name)
    plt.show(block=False)
    plt.figure()



# ###############################################
# functions for doing ranking curve #############
# API should stay consist through out this repo #
# ###############################################
# Compute the rank of the real key for a give set of predictions
def rank(predictions, plaintext_list, real_key, min_trace_idx, max_trace_idx, last_key_bytes_proba, target_byte):
    # Compute the rank
    if len(last_key_bytes_proba) == 0:
        # If this is the first rank we compute, initialize all the estimates to zero
        key_bytes_proba = np.zeros(256)
    else:
        # This is not the first rank we compute: we optimize things by using the
        # previous computations to save time!
        key_bytes_proba = last_key_bytes_proba

    for p in range(0, max_trace_idx-min_trace_idx):
        # Go back from the class to the key byte. '2' is the index of the byte (third byte) of interest.
        plaintext = plaintext_list[p][target_byte]
        for i in range(0, 256):
            # Our candidate key byte probability is the sum of the predictions logs
            # AES_Sbox[plaintext ^ i]
            tmp_label = process_data.aes_internal(plaintext, i)
            proba = predictions[p][tmp_label]
            if proba != 0:
                key_bytes_proba[i] += np.log(proba)
            else:
                # We do not want an -inf here, put a very small epsilon
                # that corresponds to a power of our min non zero proba
                min_proba_predictions = predictions[p][np.array(predictions[p]) != 0]
                if len(min_proba_predictions) == 0:
                    print("[LOG] -- Error: got a prediction with only zeroes ... this should not happen!")
                    sys.exit(-1)
                min_proba = min(min_proba_predictions)
                key_bytes_proba[i] += np.log(min_proba**2)
                '''
                min_proba = 0.000000000000000000000000000000000001
                key_bytes_proba[i] += np.log(min_proba**2)
                '''

    # Now we find where our real key candidate lies in the estimation.
    # We do this by sorting our estimates and find the rank in the sorted array.
    sorted_proba = np.array(list(map(lambda a : key_bytes_proba[a], key_bytes_proba.argsort()[::-1])))
    real_key_rank = np.where(sorted_proba == key_bytes_proba[real_key])[0][0]
    return (real_key_rank, key_bytes_proba)


def full_ranks(proba_mat, key, plaintext_attack, min_trace_idx, max_trace_idx, target_byte, rank_step):
    # Real key byte value that we will use. '2' is the index of the byte (third byte) of interest.
    real_key = key[target_byte]
    # Check for overflow
    if max_trace_idx > proba_mat.shape[0]:
        raise ValueError("Error: asked trace index %d overflows the total traces number %d" % (max_trace_idx, dataset.shape[0]))

    # Predict our probabilities
    predictions = proba_mat[min_trace_idx:max_trace_idx, :]

    index = np.arange(min_trace_idx+rank_step, max_trace_idx, rank_step)
    f_ranks = np.zeros((len(index), 2), dtype=np.uint32)
    key_bytes_proba = []
    for t, i in zip(index, range(0, len(index))):
        real_key_rank, key_bytes_proba = rank(predictions[t-rank_step:t], plaintext_attack[t-rank_step:t], real_key, t-rank_step, t, key_bytes_proba, target_byte)
        f_ranks[i] = [t - min_trace_idx, real_key_rank]
    return f_ranks


def loadData(opts, cuda):
    # Dataloader
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    input = opts.input
    target_byte = opts.target_byte
    batch_size = opts.batch_size
    attack_window = opts.attack_window
    shifted = opts.shifted

    if '' != attack_window:
        print('[LOG] -- using the self-defined attack window.')
        tmp = opts.attack_window.split('_')
        attack_window = [int(tmp[0]), int(tmp[1])]

    test_loader = data_loader.load_source(input, target_byte, attack_window, batch_size, kwargs, shifted, trace_num=0)
    return test_loader


def test_model(opts, model, target_test_loader, cuda):
    len_target_dataset = len(target_test_loader.dataset)
    key = target_test_loader.dataset.get_data_key()

    model.eval()
    test_loss = 0
    correct = 0

    count = 0
    for data, target, textin in target_test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        s_output = model(data)
        test_loss += F.nll_loss(F.log_softmax(s_output, dim=1), target, size_average=False).item()
        # for compute ranking curve
        tmp = F.softmax(s_output, dim=1)
        tmp = tmp.cpu().detach().numpy()
        if 0 == count:
            proba_mat = tmp
            textin_mat = textin
        else:
            proba_mat = np.concatenate((proba_mat, tmp), axis=0)
            textin_mat = np.concatenate((textin_mat, textin), axis=0)
        count += 1
        # for compute accuracy
        pred = s_output.max(1)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len_target_dataset
    accuracy = correct / len_target_dataset
    print('\n[LOG] -- {} set: Average loss: {:.4f}, Accuracy: ({:.4f}%)\n'.format(opts.input, test_loss, 100. * accuracy))
    return accuracy, proba_mat, textin_mat, key


def main(opts, cuda):
    model_path = opts.model_path
    clsNum = 256
    target_byte = opts.target_byte
    min_trace_idx = 0
    max_trace_idx = opts.max_traces
    rank_step = opts.rank_step

    # load model
    #model = models.RevGrad(num_classes=clsNum)
    #model.load_state_dict(torch.load(model_path))
    model = torch.load(model_path)
    model.eval()

    # load data
    test_loader = loadData(opts, cuda)

    # run test
    acc, proba_mat, textin_mat, key = test_model(opts, model, test_loader, cuda)
    print('[LOG] -- test accuracy is: {:f}'.format(acc))

    # run rank key curve function
    assert(proba_mat.shape[0] == textin_mat.shape[0])
    f_ranks = full_ranks(proba_mat, key, textin_mat, min_trace_idx, max_trace_idx, target_byte, rank_step)

    # plot_rank_curve(opts, ranks, target_byte):
    # We plot the results
    # f_ranks[i] = [t, real_key_rank]
    x = [f_ranks[i][0] for i in range(0, f_ranks.shape[0])]
    y = [f_ranks[i][1] for i in range(0, f_ranks.shape[0])]

    dataset_name = os.path.basename(opts.input).split('.')[0]
    model_file_name = os.path.basename(opts.model_path).split('.')[0]
    fig_save_dir = os.path.join(opts.output, 'figures')
    os.makedirs(fig_save_dir, exist_ok=True)
    fig_save_name = os.path.join(fig_save_dir, str(dataset_name) + '_rank_performance_byte_{}.png'.format(target_byte))

    # def plot_figure(x, y, model_file_name, dataset_name, fig_save_name):
    plot_figure(x, y, model_file_name, dataset_name, fig_save_name)
    print('[LOG] -- figure save to file: {}'.format(fig_save_name))

    outpath = os.path.join(opts.output, 'raw_data')
    os.makedirs(outpath, exist_ok=True)
    outfile = os.path.join(outpath, '{}_{}.npz'.format(dataset_name, model_file_name))
    np.savez(outfile, x=x, y=y)
    print('raw data save to path: {}'.format(outfile))

    print('[LOG] -- all done!')


def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='')
    parser.add_argument('-o', '--output', help='')
    parser.add_argument('-m', '--model_path', help='')
    parser.add_argument('-s', '--shifted', type=int, default=200, help='')
    parser.add_argument('-b', '--batch_size', type=int, default=100, help='')
    parser.add_argument('-tb', '--target_byte', type=int, default=0, help='')
    parser.add_argument('-aw', '--attack_window', default='', help='')
    parser.add_argument('-mt', '--max_traces', type=int, default=2000, help='')
    parser.add_argument('-rs', '--rank_step', type=int, default=1, help='')
    opts = parser.parse_args()
    return opts


if __name__ == '__main__':
    opts = parseArgs(sys.argv)
    # Device settings
    cuda = torch.cuda.is_available()
    if cuda:
        seed = int(time.time() // 10000)
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        torch.cuda.manual_seed(seed)
    main(opts, cuda)
