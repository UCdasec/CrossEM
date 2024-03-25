#!/usr/bin/env python3.6

import os
import sys
import argparse
import pdb
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import random
from itertools import permutations
import seaborn as sns
from sklearn.manifold import TSNE
from tensorflow.keras.models import load_model

import mytools.tools as mytools

RootDir = os.getenv('ROOT_DIR')
modelsDir = os.path.join(RootDir, 'models')
toolsDir = os.path.join(RootDir, 'tools')
sys.path.append(toolsDir)
sys.path.append(modelsDir)
import getBinFormat
from DF_model import DF
import augData
import mytools.tools as mytools

figDir = os.path.join(RootDir, 'test_figures')
os.makedirs(figDir, exist_ok=True)


# load up data
def loadData(fpath, sample_num=100):
    # Load images and corresponding labels from the text file,
    # stack them in numpy arrays and return
    if not os.path.isfile(fpath):
        raise ValueError("File path {} does not exist. Exiting...".format(fpath))

    wholePack = np.load(fpath)
    allData, allLabel = wholePack['x'], wholePack['y']
    datas, labels = mytools.limitData(allData, allLabel, sampleLimit=sample_num)

    # shuffle data
    datas, labels = mytools.shuffleData(datas, labels)

    # delete all useless data to save memory
    del wholePack, allData, allLabel

    return datas, labels


def visualize(x_data, y_data, outpath, titleStr=''):
    '''t-SNE'''
    tsne = TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(x_data)

    print("Org data dimension is {}. Embedded data dimension is {}".format(x_data.shape[-1], X_tsne.shape[-1]))

    '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(8, 8))
    #pdb.set_trace()
    #for i in range(X_norm.shape[0]):
    #    plt.text(X_norm[i, 0], X_norm[i, 1], str(y_data[i]), color=plt.cm.Set1(y_data[i]), fontdict={'weight': 'bold', 'size': 9})
    plt.scatter(X_norm[:, 0], X_norm[:, 1], c=y_data, cmap='viridis', marker='o')

    plt.xticks([])
    plt.yticks([])

    if titleStr:
        tmp = titleStr.split('_')
        titleStr = ' '.join(tmp)
        plt.title(titleStr)

    plt.savefig(outpath)
    plt.show()
    plt.close()


def createFeatModel(opts, emb_size):
    inp_shape = (opts.data_dim, 1)
    model = DF(input_shape=inp_shape, emb_size=emb_size, Classification=False)
    return model


def copyWeights(pre_model, feat_model):
    layNum = len(feat_model.layers)
    for l1, l2 in zip(pre_model.layers[:layNum], feat_model.layers[:layNum]):
        l2.set_weights(l1.get_weights())
        l2.trainable = False
    return feat_model


def computeFeatures(feat_model, x):
    feats = feat_model.predict(x)
    return feats


def main(opts):
    dataName = os.path.basename(opts.input).split('.')[0]

    if 'ori' == opts.method:
        methodName = 'original_data'
    elif 'finetune' == opts.method:
        methodName = 'finetuned_model'
        emb_size = 100
    elif 'triplet' == opts.method:
        methodName = 'triplet_model'
        emb_size = 64
    elif 'ada' == opts.method:
        methodName = 'ADA_model'
        emb_size = 512
    else:
        raise

    # show original data
    sampleNum = 50  # For speed of computation, only run on a subset
    x_data, y_data = loadData(opts.input, sampleNum)
    data_dim = opts.data_dim
    x_data = x_data[:, :data_dim]

    if 'ori' == opts.method:
        feats = x_data
    else:
        x_data = x_data[:, :, np.newaxis]

        # show triplet data
        model = load_model(opts.modelFile)
        feats = model.predict(x_data)

    outpath = os.path.join(figDir, "tsne_{}_{}_shot_{}.pdf".format(methodName, dataName, opts.nShot))
    visualize(feats, y_data, outpath)

    print('figure save to path: {}'.format(outpath))


def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='')
    parser.add_argument('-d', '--data_dim', type=int, default=5000, help='output file name')
    parser.add_argument('-ns', '--nShot', type=int, help='indicate it trained with how many shots')
    parser.add_argument('-m', '--method', help='choose from ori/finetune/triplet/ada')
    parser.add_argument('-mf', '--modelFile', help='')
    parser.add_argument('-g', '--useGpu', action='store_true', help='')
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parseArgs(sys.argv)
    if opts.useGpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    main(opts)
