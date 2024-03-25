#!/usr/bin/python3.6
import pdb
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv1D, BatchNormalization,ReLU
from tensorflow.keras.layers import GlobalMaxPool1D, Input, AveragePooling1D
from tensorflow.keras.layers import Flatten, GlobalMaxPooling1D, Dropout
from tensorflow.keras.layers import Activation, GlobalAveragePooling1D, MaxPooling1D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model, Sequential

#### MLP Best model (6 layers of 200 units)
def mlp_best(node=200,layer_nb=6,input_dim=1000):
	model = Sequential()
	model.add(Dense(node, input_dim=input_dim, activation='relu'))
	for i in range(layer_nb-2):
		model.add(Dense(node, activation='relu'))
	model.add(Dense(256, activation='softmax'))
	optimizer = RMSprop(lr=0.00001)
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model


def test():
    inp_shape = (95, 1)

    # test the cnn model
    best_model = mlp_best(inp_shape, emb_size=256, classification=True)
    best_model.summary()

if __name__ == '__main__':
    test()