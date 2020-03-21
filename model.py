from collections import namedtuple
from typing import List

from tensorflow.keras.layers import Dense, Conv2D, Flatten, BatchNormalization, ReLU, Add, Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *


from tablut import Tafl, SPACE_ACTION
from nn_input import NNInputs


Replay = namedtuple('REPLAY', ['state', 'policy', 'reward'])

def convolutional_layer(INPUT, num_filters = 256, kernel_size = (1,1)):
    layer = Conv2D(num_filters, kernel_size, data_format='channels_first')(INPUT)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    return layer

def residual_layer(INPUT):
    SHORCUT = INPUT
    layer = convolutional_layer(INPUT)
    layer = convolutional_layer(layer)
    layer = convolutional_layer(layer)
    shortcut = convolutional_layer(SHORCUT)
    return Add()([layer, shortcut])

def value_head(model):
    model = convolutional_layer(model, 1)
    model = Dense(256)(model)
    model = ReLU()(model)
    model = Flatten()(model)
    model = Dense(1, activation='tanh')(model)
    return model


def policy_head(model):
    model = Conv2D(64, (1,1))(model)
    model = BatchNormalization()(model)
    model = ReLU()(model)
    model = Flatten()(model)
    model = Dense(256)(model)
    model = ReLU()(model)
    model = Dense(len(SPACE_ACTION))(model)
    return model


def gen_model() -> Model:
    input = Input(batch_shape=(1,21,9,9))
    common = convolutional_layer(input)
    common = residual_layer(common)
    common = residual_layer(common)

    vhead = value_head(common)
    phead = policy_head(common)

    model = Model(inputs=input, outputs=[vhead, phead])
    model.compile(loss='mean_squared_error', optimizer='sgd')
    return model

model = gen_model()
env = Tafl()


def predict(env: Tafl, nn: Model):
    input = NNInputs.from_Tafl(env).to_neural_input()
    input = input.reshape((1, 21, 9, 9))
    value, policy = nn.predict(input)
    outp = dict()
    for indx in env.mask:
        outp[SPACE_ACTION[indx]] = policy[0][indx]
    return value, outp



