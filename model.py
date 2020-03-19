from keras import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten
from tablut import Tafl, SPACE_ACTION
from nn_input import NNInputs
import numpy as np

def model() -> Model:
    model = Sequential()
    model.add(Conv2D(162, kernel_size=3,
                     strides=(2,2),
                     activation='relu',
                     input_shape=(21,9,9),
                     data_format='channels_first'))
    model.add(Conv2D(324, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(len(SPACE_ACTION), activation='softmax'))
    model.compile(loss='mean_squared_error', optimizer='sgd')
    return model

model = model()
env = Tafl()
print(env)
from pprint import pprint as _print
for e in range(1):
    env.reset()
    input = NNInputs.from_Tafl(env).to_neural_input()
    for time in range(500):
        prediction = model.predict(input)
        len(prediction)


