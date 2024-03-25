#!/usr/bin/python3.6
import pdb
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv1D, BatchNormalization
from tensorflow.keras.layers import GlobalMaxPool1D, Input, AveragePooling1D
from tensorflow.keras.layers import Flatten, GlobalMaxPooling1D, Dropout
from tensorflow.keras.layers import Activation, GlobalAveragePooling1D, MaxPooling1D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model, Sequential
#purdue stuff
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import ReLU, BatchNormalization, Dropout



### CNN Best model
def cnn_best_norm(input_shape, emb_size=256, classification=True, compile=True):
    inp = Input(shape=input_shape)
    # Block 1
    x = Conv1D(64, 11, strides=2, activation='relu', padding='same', name='block1_conv1')(inp)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)
    x = BatchNormalization()(x)
    # Block 2
    x = Conv1D(128, 11, activation='relu', padding='same', name='block2_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block2_pool')(x)
    x = BatchNormalization()(x)
    # Block 3
    x = Conv1D(256, 11, activation='relu', padding='same', name='block3_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block3_pool')(x)
    x = BatchNormalization()(x)
    # Block 4
    x = Conv1D(512, 11, activation='relu', padding='same', name='block4_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block4_pool')(x)
    x = BatchNormalization()(x)
    # Block 5
    x = Conv1D(512, 11, activation='relu', padding='same', name='block5_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block5_pool')(x)
    x = BatchNormalization()(x)
    # Classification block

    x = Flatten(name='block_flatten')(x)

    x = Dense(4096, activation='relu', name='block_fc1')(x)
    x = Dense(4096, activation='relu', name='block_fc2')(x)

    if classification:
        x = Dense(emb_size, activation='softmax', name='preds')(x)
        # Create model.
        model = Model(inp, x, name='cnn_best_norm')
        if compile:
            optimizer = RMSprop(lr=0.00001)
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
            print('[log] --- finish construct the cnn_best_norm model')
        return model
    else:
        return inp, x

def resnet_v1(input_shape, emb_size=256, without_permind=0):
    depth = 19
    if (depth - 1) % 18 != 0:
        raise ValueError('depth should be 18n+1 (e.g., 19, 37, 55 ...)')
    
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 1) / 18)
    inputs = Input(shape=(1000,))
    x = resnet_layer(inputs=inputs)
    
    # Instantiate the stack of residual units
    for stack in range(9):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:
                strides = 2
            y = resnet_layer(inputs=x, num_filters=num_filters, strides=strides, activation=None)
            if stack > 0 and res_block == 0:
                x = resnet_layer(inputs=x, num_filters=num_filters, kernel_size=1, strides=strides, activation=None, batch_normalization=False)
            x = add([x, y])
            x = Activation('relu')(x)
        if (num_filters < 256):
            num_filters *= 2
    
    x = AveragePooling1D(pool_size=4)(x)
    x = Flatten()(x)
    x_alpha = alpha_branch(x)
    x_beta = beta_branch(x)
    x_sbox_l = []
    x_permind_l = []
    
    for i in range(16):
        x_sbox_l.append(sbox_branch(x, i))
        x_permind_l.append(permind_branch(x, i))
    
    if without_permind != 1:
        model = Model(inputs, [x_alpha, x_beta] + x_sbox_l + x_permind_l, name='extract_resnet')
    else:
        model = Model(inputs, [x_alpha, x_beta] + x_sbox_l, name='extract_resnet_without_permind')
    
    optimizer = Adam()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model

### CNN Best model
def cnn_best(input_shape, emb_size=256, classification=True, compile=True):
    inp = Input(shape=input_shape)
    # Block 1
    x = Conv1D(64, 11, strides=2, activation='relu', padding='same', name='block1_conv1')(inp)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)
    # Block 2
    x = Conv1D(128, 11, activation='relu', padding='same', name='block2_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block2_pool')(x)
    # Block 3
    x = Conv1D(256, 11, activation='relu', padding='same', name='block3_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block3_pool')(x)
    # Block 4
    x = Conv1D(512, 11, activation='relu', padding='same', name='block4_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block4_pool')(x)
    # Block 5
    x = Conv1D(512, 11, activation='relu', padding='same', name='block5_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block5_pool')(x)
    # Classification block

    x = Flatten(name='block_flatten')(x)

    x = Dense(4096, activation='relu', name='block_fc1')(x)
    x = Dense(4096, activation='relu', name='block_fc2')(x)

    if classification:
        x = Dense(emb_size, activation='softmax', name='preds')(x)
        # Create model.
        model = Model(inp, x, name='cnn_best')
        if compile:
            optimizer = RMSprop(lr=0.00001)
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
            print('[log] --- finish construct the cnn_best model')
        return model
    else:
        return inp, x


def test():
    inp_shape = (95, 1)

    # test the cnn model
    best_model = cnn_best(inp_shape, emb_size=256, classification=True)
    best_model.summary()

    # test the cnn2 model
    best_model = cnn_best(inp_shape, emb_size=256, classification=True)
    best_model.summary()

    # test the hamming weight model
    model = cnn_best_norm(input_shape=inp_shape, emb_size=9, classification=True)
    model.summary()

    # test the hamming weight model
    model = cnn_best(input_shape=inp_shape, emb_size=9, classification=True)
    model.summary()


if __name__ == '__main__':
    test()
