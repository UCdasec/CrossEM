import os
import sys
import argparse
import pdb
import h5py
import time

import tensorflow as tf
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
sys.path.append('tools')
import loadData
import model_zoo
import checking_tool
import csv


def print_run_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        print('[LOG -- RUN TIME] -- current function [{}] run time is {:f}'.format(func.__name__, end-start))
    return wrapper



def load_data(opts, forceKey=None):
    '''data loading function'''
    target_byte = opts.target_byte
    leakage_model = opts.leakage_model
    data_path = opts.input
    trace_num =opts.trace_num
    method = opts.preprocess
    attack_window = opts.attack_window
    whole_pack = np.load(data_path)
    traces, text_in, key = loadData.load_data_base(whole_pack, attack_window, method, trace_num=trace_num, shifted=0)

    print("LABELS GENERATED USING KEY",forceKey)
    labels = loadData.get_labels(text_in,forceKey, target_byte, leakage_model)
    inp_shape = (traces.shape[1], 1)
    loadData.data_info(traces.shape, text_in.shape, key)
    clsNum = 9 if 'HW' == leakage_model else 256
    print('[LOG] -- class number is: ', clsNum)
    labels = to_categorical(labels, clsNum)

    return traces, labels, inp_shape, clsNum





def fine_tune(opts,modelDir, X_profiling, Y_profiling, model, epochs,byte_n,mainAcc, mainLoss, batch_size=100, verbose=False,non_fixed=2):
    its=opts.test_num
    itn=opts.trace_num-its
    print(" LOG----- NUmber of layers to fine tune",non_fixed)
    model_save_file = os.path.join(modelDir,str(byte_n), 'best_model.h5')

    # Save model every epoch
    checkpointer = ModelCheckpoint(model_save_file, monitor='val_accuracy', verbose=verbose, save_best_only=True, mode='max')
    callbacks = [checkpointer]

    depth = len(model.layers)
    for i in range(depth - non_fixed):
        model.layers[i].trainable = False
    # Get the input layer shape and Sanity check
    model.summary()
    input_layer_shape = model.get_layer(index=0).input_shape
    if isinstance(input_layer_shape, list):
        input_layer_shape = input_layer_shape[0]
    Reshaped_X_profiling = loadData.sanity_check(input_layer_shape, X_profiling)

    hist = model.fit(x=Reshaped_X_profiling[0:itn], y=Y_profiling[0:itn],
                     validation_split=0.1, batch_size=batch_size,
                     verbose=verbose, epochs=epochs,
                     shuffle=True)


    # run the accuracy test
    loss, acc  = model.evaluate(Reshaped_X_profiling, Y_profiling, verbose=opts.verbose)
    mainAcc.append(acc)
    mainLoss.append(loss)

    return acc,loss

@print_run_time
def main(opts):
    # get the params
    leakage_model = opts.leakage_model
    verbose = opts.verbose
    epochs = opts.epochs
    batch_size = 100
    modelDir = os.path.join(opts.output, 'model')
    modelDirft = os.path.join(opts.output, 'finetuning/model')
    mainAcc=[]
    mainLoss=[]
    # Start non-profiling loop
    totalTrainTimeStart = time.time()
    # Check if directory exists, if not, create it
    if not os.path.exists(opts.output):
        os.makedirs(opts.output)
    else:
        print("DSDASDDSADASDASDS")
    for i in range(255):

        X_profiling_ft, Y_profiling_ft, input_shape_ft, clsNum_ft = load_data(opts,forceKey=i)
        print('[LOG] -- trace data shape is: ', X_profiling_ft.shape)
        cnn_model_path = os.path.join(opts.model_dir, 'model', 'best_model.h5')
        # Load model
        best_model = checking_tool.load_best_model(cnn_model_path)
        best_model.summary()
        print("...................................................................PRETRAINED MODEL LOADED.......................................................................")
        acc,loss=fine_tune(opts,modelDirft, X_profiling_ft, Y_profiling_ft, best_model, epochs,i,mainAcc,mainLoss, batch_size, verbose,opts.non_fixed)

        print('[LOG] -- test acc is:',(mainAcc[i]))
        print('[LOG] -- loss is:',(mainLoss[i]))
    
    file_name=os.path.join(opts.output, 'acc.csv')
    # Open the CSV file in write mode
    with open(file_name, 'w', newline='') as csvfile:
        # Create a CSV writer object
        csv_writer = csv.writer(csvfile)

        # Write the header
        csv_writer.writerow(['Accuracy', 'Loss'])

        # Write accuracy and loss for each iteration
        for acc, loss in zip(mainAcc, mainLoss):
            csv_writer.writerow([acc, loss])
    print(f"The list has been saved to {file_name}")


def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='')
    parser.add_argument('-tn', '--trace_num', type=int, default=15000, help='')
    parser.add_argument('-ts', '--test_num', type=int, default=5000, help='')
    parser.add_argument('-o', '--output', help='')
    parser.add_argument('-aw', '--attack_window', default='', help='overwrite the attack window')
    parser.add_argument('-m', '--model_dir', help='')
    parser.add_argument('-tb', '--target_byte', type=int, default=2, help='default value is 2')

    parser.add_argument('-v', '--verbose', action='store_true', help='')
    parser.add_argument('-lm', '--leakage_model', choices={'HW', 'ID'}, help='')
    parser.add_argument('-e', '--epochs', type=int, default=150, help='')
    parser.add_argument('-ftl', '--non_fixed', type=int, default=1, help='Number of layers to finetune')

    parser.add_argument('-pp', '--preprocess', default='', choices={'norm', 'scaling', ''}, help='')
    parser.add_argument('-sh', '--shifted', type=int, default=0, help='')

    opts = parser.parse_args()
    return opts

if __name__ == "__main__":
    opts = parseArgs(sys.argv)
    if tf.test.is_gpu_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    main(opts)
