#!/usr/bin/python3.6
import pdb
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv1D, BatchNormalization,ReLU
from tensorflow.keras.layers import GlobalMaxPool1D, Input, AveragePooling1D
from tensorflow.keras.layers import Flatten, GlobalMaxPooling1D, Dropout
from tensorflow.keras.layers import Activation, GlobalAveragePooling1D, MaxPooling1D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model, Sequential


### CNN Best model
def cnn_best(input_shape, emb_size=2, classification=True, compile=True):
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

def dnn_model(input_shape, emb_size=256, classification=True, compile=True):
    inp = Input(shape=input_shape, name='input')
    
    x = Dense(3000, name='dense')(inp)
    x = ReLU(name='re_lu')(x)
    x = BatchNormalization(name='batch_normalization')(x)
    x = Dropout(0.5, name='dropout')(x)

    x = Dense(100, name='dense_1')(x)
    x = ReLU(name='re_lu_1')(x)
    x = BatchNormalization(name='batch_normalization_1')(x)
    x = Dropout(0.5, name='dropout_1')(x)

    x = Dense(1024, name='dense_2')(x)
    x = ReLU(name='re_lu_2')(x)
    x = BatchNormalization(name='batch_normalization_2')(x)
    x = Dropout(0.5, name='dropout_2')(x)

    x = Dense(512, name='dense_3')(x)
    x = ReLU(name='re_lu_3')(x)
    x = BatchNormalization(name='batch_normalization_3')(x)
    x = Dropout(0.5, name='dropout_3')(x)

    x = Flatten(name='flatten')(x)

    x = Dense(256, name='dense_4')(x)

    model = Model(inputs=inp, outputs=x, name='dnn_model')
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
    return model


#### MLP Best model (6 layers of 200 units)
def mlp_best(node=200,layer_nb=6,input_dim=1000):
	model = Sequential()
	model.add(Dense(node, input_dim=input_dim, activation='relu'))
	for i in range(layer_nb-2):
		model.add(Dense(node, activation='relu'))
	model.add(Dense(2, activation='softmax'))
	optimizer = RMSprop(lr=0.00001)
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model
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