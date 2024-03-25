#!/usr/bin/python3
import os
import sys
import argparse
import pdb
import h5py
import statistics
import math

import numpy as np
from collections import defaultdict
import ast
import matplotlib.pyplot as plt

import mytools.draw_base as draw_base


'''
in this file we gona compare two ranking curve and plot them in same figure
'''

def load_data(opts):
    ''' load all the pre-calculate data and draw figures '''
    dpath_1 = opts.dpath_1
    dpath_2 = opts.dpath_2

    dpack1 = np.load(dpath_1)
    dpack2 = np.load(dpath_2)

    x1, y1 = dpack1['x'], dpack1['y']
    x2, y2 = dpack2['x'], dpack2['y']

    min_val = min(len(x1), len(x2))
    x1 = x1[:min_val]
    x2 = x2[:min_val]
    y1 = y1[:min_val]
    y2 = y2[:min_val]
    return x1, y1, x2, y2


def plot_figure(all_x, all_y, labels, fig_save_name, marker_num, marker_size):
    xlabel = 'No. of test traces'
    ylabel = 'Mean rank'
    showGrid = False

    xTicks, yTicks = 0, [0, 64, 128, 192, 256]
    xLim, yLim = 0, [0, 256]

    data = draw_base.DataArray(all_x, all_y)

    # default value: LABELSIZE=26, LEGENDSIZE=26, TICKSIZE=26, LWidth=2
    LABELSIZE = 28
    LEGENDSIZE = 26
    TICKSIZE = 26
    LWidth = 2

    label1 = labels[0]
    label2 = labels[1]
    label3 = labels[2]
    label4 = labels[3]
    label5 = labels[4]

    context = draw_base.Context(xlabel, ylabel, file2save=fig_save_name, y1Lim=yLim, xLim=xLim,
                                label1=label1, label2=label2, label3=label3, label4=label4,
                                label5=label5, xTicks=xTicks, y1Ticks=yTicks, LWidth=LWidth,
                                showGrid=showGrid, MSize=marker_size, marker_num=marker_num,
                                LABELSIZE=LABELSIZE, LEGENDSIZE=LEGENDSIZE, TICKSIZE=TICKSIZE)

    draw_base.draw_results(data, 1, context)


# Check a saved model against one of the testing databases Attack traces
def main(opts):
    # prepare the saving path
    outDir = opts.output
    os.makedirs(outDir, exist_ok=True)
    fig_name = '{}_vs_{}'.format(os.path.basename(opts.dpath_1).split('.')[0], os.path.basename(opts.dpath_2).split('.')[0])
    fig_save_name = os.path.join(opts.output, '{}.png'.format(fig_name))
    print('figure save to file: {}'.format(fig_save_name))

    max_points = opts.max_points
    x1, y1, x2, y2 = load_data(opts)
    x1, x2 = x1[:max_points], x2[:max_points]
    y1, y2 = y1[:max_points], y2[:max_points]

    labels = ['CNN', 'TripletPower (Ours)', '', '', '']

    all_x = [x1, x2]
    all_y = [y1, y2]
    plot_figure(all_x, all_y, labels, fig_save_name, opts.marker_num, opts.marker_size)

    print('[LOG] --  all done!')


def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-d1', '--dpath_1', help='')
    parser.add_argument('-d2', '--dpath_2', help='')
    parser.add_argument('-o', '--output', help='')
    parser.add_argument('-mp', '--max_points', type=int, default=0, help='')
    parser.add_argument('-mn', '--marker_num', type=int, default=3, help='')
    parser.add_argument('-ms', '--marker_size', type=int, default=8, help='')
    opts = parser.parse_args()
    return opts


if __name__=="__main__":
    opts = parseArgs(sys.argv)
    main(opts)
