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

sys.path.append("tools")
import data_ada
import models
import key_rank_new as key_rank


def loadData(opts, cuda):
    # Dataloader
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    input = opts.input
    target_byte = opts.target_byte
    batch_size = opts.batch_size
    attack_window = opts.attack_window
    shifted = opts.shifted
    leakage_model = opts.leakage_model      # 'ID'
    trace_num = opts.trace_num      # 20000

    test_loader, inp_shape = data_ada.test_loader(input, target_byte, attack_window, batch_size, trace_num, shifted, leakage_model, kwargs)
    #test_loader, inp_shape = data_ada.loader(input, target_byte, attack_window, batch_size, trace_num, shifted, leakage_model, kwargs)
    return test_loader, inp_shape


def test_model(opts, model, target_test_loader, cuda):
    len_target_dataset = len(target_test_loader.dataset)
    key = target_test_loader.dataset.get_data_key()

    model.eval()

    test_loss = 0
    correct = 0
    count = 0
    for data, target, textin in target_test_loader:
        data, target = Variable(data), Variable(target)
        if cuda:
            data, target = data.float().cuda(), target.cuda()

        _, s_output, t_output = model(data, alpha=0)
        softmax_out = F.softmax(s_output, dim=1)
        test_loss += F.nll_loss(softmax_out, target, size_average=False).item()
        # for compute accuracy
        pred = s_output.max(1)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        # for compute ranking curve
        # tmp = F.softmax(s_output, dim=1)
        tmp = softmax_out.cpu().detach().numpy()
        if 0 == count:
            proba_mat = tmp
            textin_mat = textin
        else:
            proba_mat = np.concatenate((proba_mat, tmp), axis=0)
            textin_mat = np.concatenate((textin_mat, textin), axis=0)
        count += 1

    test_loss /= len_target_dataset
    accuracy = correct / len_target_dataset
    dataset = os.path.basename(opts.input)
    print('[LOG] -- \n{} set: Average loss: {:f}, Accuracy: {:f}'.format(dataset, test_loss, accuracy))
    return accuracy, proba_mat, textin_mat, key


def main(opts, cuda):
    modelDir = os.path.join(opts.output, 'model')
    rank_dir = os.path.join(opts.output, 'rank_dir')
    os.makedirs(rank_dir, exist_ok=True)
    leakage_model = opts.leakage_model

    clsNum = 9 if 'HW' == leakage_model else 256
    batch_size = 100
    target_byte = opts.target_byte

    # load data
    test_loader, inp_shape = loadData(opts, cuda)

    # load model
    model_path = os.path.join(modelDir, 'final_model.pkl')
    model = models.RevGrad(inp_shape, num_classes=clsNum, batch_size=batch_size)

    model_dict = torch.load(model_path)
    model.load_state_dict(model_dict)
    #model = torch.load(model_path)
    if cuda:
        model.cuda()
    model.eval()

    # run test
    acc, proba_mat, textin_mat, key = test_model(opts, model, test_loader, cuda)
    print('[LOG] -- test accuracy is: {:f}'.format(acc))

    #data_ada.ranking_curve(opts, proba_mat, textin_mat, key, target_byte, rank_dir)
    max_trace_num = 5000
    key_rank.ranking_curve(proba_mat, key, textin_mat, target_byte, rank_dir, leakage_model, max_trace_num)
    print('[LOG] -- all done!')


def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='')
    parser.add_argument('-o', '--output', help='')
    parser.add_argument('-b', '--batch_size', type=int, default=100, help='')
    parser.add_argument('-lm', '--leakage_model', choices={'HW', 'ID'}, help='')
    parser.add_argument('-sh', '--shifted', type=int, default=0, help='')
    parser.add_argument('-tb', '--target_byte', type=int, default=0, help='')
    parser.add_argument('-aw', '--attack_window', default='', help='')
    parser.add_argument('-tn', '--trace_num', type=int, default=2000, help='')
    parser.add_argument('-rs', '--rank_step', type=int, default=5, help='')
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
