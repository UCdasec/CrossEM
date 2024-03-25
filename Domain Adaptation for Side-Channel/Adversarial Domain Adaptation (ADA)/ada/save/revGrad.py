import os
import sys
import math
import argparse
import numpy as np
import time
import pdb
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import model_zoo
from tqdm import tqdm

sys.path.append("tools")
import data_ada
import models as models


def plot_figure(x, y, fig_save_path, title_str, xlabel, ylabel):
    plt.title(title_str)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.plot(x, y)
    plt.savefig(fig_save_path)
    plt.show(block=False)


def loadData(opts, batch_size, cuda):
    # Dataloader
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    source = opts.source
    target = opts.target
    target_byte = opts.target_byte

    src_attack_window = opts.src_attack_window
    tar_attack_window = opts.tar_attack_window
    shifted = opts.shifted
    leakage_model = opts.leakage_model

    trace_num_src = opts.trace_num_src     # 40000
    trace_num_tar = opts.trace_num_tar     # 20000
    source_loader, inp_shape = data_ada.loader(source, target_byte, src_attack_window, batch_size, trace_num_src, 0, leakage_model, kwargs)
    target_loader, inp_shape = data_ada.loader(target, target_byte, tar_attack_window, batch_size, trace_num_tar, shifted, leakage_model, kwargs)
    return source_loader, target_loader, inp_shape


def adjust_learning_rate(optimizer):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    param_group = optimizer.param_groups[0]
    lr = param_group['lr']
    tmp_lr = lr / 2
    param_group['lr'] = tmp_lr
    print('[LOG] -- learning rate adjust from {:f} to {:f}'.format(lr, tmp_lr))


# For every epoch training
def train(opts, epoch, model, source_loader, target_loader, batch_size, learning_rate, cuda):
    ''' training model '''
    len_source_loader = len(source_loader)
    len_target_loader = len(target_loader)
    len_source_dataset = len(source_loader.dataset)

    #optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    loss_class = torch.nn.CrossEntropyLoss()
    loss_domain = torch.nn.CrossEntropyLoss()

    data_source_iter = iter(source_loader)
    data_target_iter = iter(target_loader)
    dlabel_src = Variable(torch.ones(batch_size).long().cuda())
    dlabel_tgt = Variable(torch.zeros(batch_size).long().cuda())

    #if (epoch % 30) == 0:
    #    adjust_learning_rate(optimizer)

    for i in range(1, len_source_loader+1):
        model.train()

        # the parameter for reversing gradients
        p = float(i + epoch * len_source_loader) / opts.epochs / len_source_loader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        #alpha = 1

        # for the source domain batch
        source_data, source_label, textin = next(data_source_iter)
        source_data, source_label = Variable(source_data), Variable(source_label)
        if cuda:
            source_data, source_label = source_data.float().cuda(), source_label.cuda()

        _, clabel_src, dlabel_pred_src = model(source_data, alpha=alpha)
        label_loss = loss_class(clabel_src, source_label)
        domain_loss_src = loss_domain(dlabel_pred_src, dlabel_src)

        # for the target domain batch
        target_data, target_label, textin = next(data_target_iter)
        if i % len_target_loader == 0:
            data_target_iter = iter(target_loader)

        target_data = Variable(target_data)
        if cuda:
            target_data, target_label = target_data.float().cuda(), target_label.cuda()

        _, clabel_tgt, dlabel_pred_tgt = model(target_data, alpha=alpha)
        domain_loss_tgt = loss_domain(dlabel_pred_tgt, dlabel_tgt)

        domain_loss_total = domain_loss_src + domain_loss_tgt
        loss_total = label_loss + domain_loss_total

        optimizer.zero_grad()
        # label_loss.backward()
        loss_total.backward()
        optimizer.step()

        epoch_format_str = '[LOG] -- Train Epoch: {} [{}/{} ({:.0f}%)]\tlabel_Loss: {:.6f}\tdomain_Loss: {:.6f}'
        if i % 100 == 0:
            print(epoch_format_str.format(epoch, i * len(source_data), len_source_dataset,
                  100. * i / len_source_loader, label_loss.item(), domain_loss_total.item()))



# For every epoch evaluation
def test(opts, model, target_test_loader, cuda):
    len_target_dataset = len(target_test_loader.dataset)

    model.eval()

    test_loss = 0
    correct = 0
    for data, target, textin in target_test_loader:
        data, target = Variable(data), Variable(target)
        if cuda:
            data, target = data.float().cuda(), target.cuda()

        _, s_output, t_output = model(data, alpha=0)
        test_loss += F.nll_loss(F.log_softmax(s_output, dim=1), target, size_average=False).item()
        pred = s_output.max(1)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len_target_dataset
    accuracy = correct / len_target_dataset
    test_format_str = '\n[LOG] -- {} set: Average loss: {:f}, Accuracy: {:f}'
    print(test_format_str.format(opts.source, test_loss, accuracy))
    return test_loss, accuracy


def main(opts, cuda):
    # set params
    clsNum = 9 if 'HW' == opts.leakage_model else 256
    batch_size = 100
    learning_rate = 0.000005
    modelDir = os.path.join(opts.output, 'model')
    os.makedirs(modelDir, exist_ok=True)

    # load the data
    source_loader, target_loader, inp_shape = loadData(opts, batch_size, cuda)

    model = models.RevGrad(inp_shape, batch_size, num_classes=clsNum)
    print(model)
    if cuda:
        model.cuda()

    # start training
    loss_list = []
    start = time.time()
    for epoch in range(1, opts.epochs+1):
        train(opts, epoch, model, source_loader, target_loader, batch_size, learning_rate, cuda)
        # now test on target, test for every epoch
        test_loss, t_acc = test(opts, model, source_loader, cuda)
        print('[LOG] -- source domain accuracy is: ', t_acc.item())
        loss_list.append(test_loss)

    end = time.time()
    last_time = end - start
    print('[LOG] --- training time is: ', last_time)

    # only save the parameters
    model_save_path = os.path.join(modelDir, 'final_model.pkl')
    torch.save(model.state_dict(), model_save_path)
    print('[LOG] -- training finished, model save to path: {}'.format(model_save_path))

    # print the loss figure on source domain data
    x = list(range(1, len(loss_list)+1))
    fig_save_path = os.path.join(modelDir, 'loss.png')
    title = 'loss curve'
    xlabel = 'loss'
    ylabel = 'epoch'
    plot_figure(x, loss_list, fig_save_path, title, xlabel, ylabel)


def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', help='the source domain')
    parser.add_argument('-t', '--target', help='the target domain')
    parser.add_argument('-o', '--output', help='model save models')
    parser.add_argument('-tb', '--target_byte', type=int, default=0, help='')
    parser.add_argument('-sh', '--shifted', type=int, default=0, help='')
    parser.add_argument('-saw', '--src_attack_window', default='', help='')
    parser.add_argument('-taw', '--tar_attack_window', default='', help='')
    parser.add_argument('-srn', '--trace_num_src', type=int, help='')
    parser.add_argument('-trn', '--trace_num_tar', type=int, help='')
    parser.add_argument('-lm', '--leakage_model', choices={'HW', 'ID'}, help='')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='the number of epochs')

    opt = parser.parse_args()
    print(opt)
    return opt


if __name__ == "__main__":
    opts = parseArgs(sys.argv)
    # Device settings
    cuda = torch.cuda.is_available()
    if cuda:
        seed = int(time.time() // 10000)
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        torch.cuda.manual_seed(seed)
    main(opts, cuda)
