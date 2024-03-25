import os
import sys
import pdb
import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

sys.path.append("tools")
import data_ada
import models
import key_rank


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


def loadData(opts, cuda):
    # Dataloader
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    tune_data = opts.tune_data
    test_data = opts.test_data

    target_byte = opts.target_byte
    batch_size = 100
    attack_window = opts.attack_window
    shifted = opts.shifted
    leakage_model = opts.leakage_model
    trace_num = opts.trace_num      # 20000
    tune_num = 10000

    tune_loader, inp_shape = data_ada.loader(tune_data, target_byte, attack_window, batch_size, tune_num, shifted, leakage_model, kwargs)
    test_loader, inp_shape = data_ada.loader(test_data, target_byte, attack_window, batch_size, trace_num, shifted, leakage_model, kwargs)
    return tune_loader, test_loader, inp_shape


# model = tune_model(opts, src_model, tune_loader, inp_shape, batch_size, tune_modelDir, cuda)
def tune_model(opts, src_model, tune_loader, inp_shape, batch_size, tune_modelDir, guess_key, cuda):
    ''' tune the last few layers '''
    clsNum = 9 if 'HW' == opts.leakage_model else 256
    batch_size = 100
    epochs = 100
    target_model = models.CNN_Best(inp_shape, num_classes=clsNum, batch_size=batch_size)

    # copy the weights
    model_dict = target_model.state_dict()
    pretrained_dict = src_model
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    target_model.load_state_dict(model_dict)

    if cuda:
        target_model.cuda()

    # tune the model
    len_tune_loader = len(tune_loader)
    len_tune_dataset = len(tune_loader.dataset)

    #optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr)
    optimizer = optim.RMSprop(target_model.parameters(), lr=0.00001)
    loss_class = torch.nn.CrossEntropyLoss()

    for e in range(1, epochs+1):
        i = 1
        data_iter = iter(tune_loader)
        while i <= len_tune_loader:
            target_model.train()

            # for the source domain batch
            inp_data, inp_label, textin = data_iter.next()
            inp_data, inp_label = Variable(inp_data), Variable(inp_label)
            if cuda:
                inp_data, inp_label = inp_data.cuda(), inp_label.cuda()

            model_pred = target_model(inp_data)
            label_loss = loss_class(model_pred, inp_label)

            optimizer.zero_grad()
            label_loss.backward()
            optimizer.step()

            out_template = '[LOG] -- Train Epoch: {} [{}/{} ({:.0f}%)]\tlabel_Loss: {:.6f}'
            if i % 100 == 0:
                print(out_template.format(e, i*len(inp_data), len_tune_dataset, 100. * i / len_tune_loader, label_loss.item()))
            i = i + 1

    #tune_model_path = os.path.join(tune_modelDir, 'guess_key_{}.pkl'.format(guess_key))
    #target_model.save(target_model.state_dict, tune_model_path)
    return target_model


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

        s_output = model(data)
        softmax_out = F.log_softmax(s_output, dim=1)
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
    dataset = os.path.basename(opts.test_data)
    print('[LOG] -- \n{} set: Average loss: {:f}, Accuracy: {:f}'.format(dataset, test_loss, accuracy))
    return accuracy, proba_mat, textin_mat, key


def main(opts, cuda):
    modelDir = os.path.join(opts.output, 'model')
    tune_modelDir = os.path.join(opts.output, 'tune_models')
    rank_dir = os.path.join(opts.output, 'rank_dir')
    os.makedirs(rank_dir, exist_ok=True)
    leakage_model = opts.leakage_model
    target_byte = opts.target_byte
    batch_size = 100

    # load data
    tune_loader, test_loader, inp_shape = loadData(opts, cuda)

    # load model
    model_path = os.path.join(modelDir, 'final_model.pkl')
    src_model_dict = torch.load(model_path)
    # model.eval()

    guess_key = tune_loader.dataset.get_data_key()[target_byte]
    model = tune_model(opts, src_model_dict, tune_loader, inp_shape, batch_size, tune_modelDir, guess_key, cuda)

    # run test
    acc, proba_mat, textin_mat, key = test_model(opts, model, test_loader, cuda)
    print('[LOG] -- test accuracy is: {:f}'.format(acc))

    max_trace_num = 500
    key_rank.ranking_curve(proba_mat, key, textin_mat, target_byte, rank_dir, leakage_model, max_trace_num)
    print('[LOG] -- all done!')


def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-tu', '--tune_data', help='')
    parser.add_argument('-te', '--test_data', help='')
    parser.add_argument('-o', '--output', help='')
    parser.add_argument('-m', '--model_path', help='')
    parser.add_argument('-lm', '--leakage_model', choices={'HW', 'ID'}, help='')
    parser.add_argument('-sh', '--shifted', type=int, default=0, help='')
    parser.add_argument('-tb', '--target_byte', type=int, default=0, help='')
    parser.add_argument('-aw', '--attack_window', default='', help='')
    parser.add_argument('-tn', '--trace_num', type=int, default=2000, help='')
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
