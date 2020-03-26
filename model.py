from collections import namedtuple
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tablut import SPACE_ACTION
import os

Replay = namedtuple('REPLAY', ['state', 'policy', 'reward'])


def convolutional_layer(INPUT, num_filters=256, kernel_size=(1, 1)):
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
    model = Conv2D(64, (1, 1))(model)
    model = BatchNormalization()(model)
    model = ReLU()(model)
    model = Flatten()(model)
    model = Dense(256)(model)
    model = ReLU()(model)
    model = Dense(len(SPACE_ACTION))(model)
    return model


def gen_model() -> Model:
    input = Input(batch_shape=(None, 23, 9, 9))
    common = convolutional_layer(input)
    common = residual_layer(common)
    common = residual_layer(common)

    vhead = value_head(common)
    phead = policy_head(common)

    model = Model(inputs=input, outputs=[vhead, phead])
    model.compile(loss='mean_squared_error', optimizer='sgd')
    return model

def load_model(filepath: str) -> Model:
    model = gen_model()
    if os.path.exists(filepath):
        print('Weights found')
        model.load_weights(filepath)
    return model

def save_weights(model: Model, filepath: str) -> bool:
    if model.save_weights(filepath):
        return True
    return False
