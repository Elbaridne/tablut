from collections import namedtuple
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tablut import SPACE_ACTION
import os

Replay = namedtuple('REPLAY', ['state', 'policy', 'reward'])


def convolutional_layer(INPUT, num_filters=512, kernel_size=(3, 3)):
    layer = Conv2D(num_filters, kernel_size, data_format='channels_first')(INPUT)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    return layer


def residual_layer(INPUT, num_filters=512, n_conv_layers=3):
    SHORCUT = INPUT
    layer = INPUT
    for i in range(n_conv_layers):
        layer = convolutional_layer(layer, num_filters)

    layer = convolutional_layer(layer)
    shortcut = convolutional_layer(SHORCUT)
    return Add()([layer, shortcut])


def value_head(model):
    model = convolutional_layer(model, 1)
    model = Dense(256)(model)
    model = ReLU()(model)
    model = Flatten()(model)
    model = Dense(1, activation='tanh', name='Value_Output')(model)
    return model


def policy_head(model):
    model = Conv2D(64, (1, 1), )(model)
    model = BatchNormalization()(model)
    model = ReLU()(model)
    model = Flatten()(model)
    model = Dense(256)(model)
    model = ReLU()(model)
    model = Dense(len(SPACE_ACTION), name='Policy_Output')(model)
    return model

def gen_model_V2() -> Model:
    input = Input(batch_shape=(None, 23, 9, 9))
    common = convolutional_layer(input, num_filters=512, kernel_size=(3,3))
    common = convolutional_layer(common, num_filters=512, kernel_size=(3, 3))
    common = convolutional_layer(common, num_filters=512, kernel_size=(1, 1))
    common = Flatten()(common)
    common = Dense(1024)(common)
    vhead = Dense(1, activation='tanh', name='Value_Output')(common)
    phead = Dense(1024)(common)
    phead = Dense(len(SPACE_ACTION), name='Policy_Output')(phead)
    losses = {'Policy_Output': 'categorical_crossentropy',
              'Value_Output': 'mean_squared_error'}

    loss_weights = {
        'Policy_Output' : 1,
        'Value_Output': 1
    }

    model = Model(inputs=input, outputs=[vhead, phead])
    model.compile(loss=losses, loss_weights=loss_weights,
                  optimizer='adam')
    return model


def gen_model() -> Model:
    input = Input(batch_shape=(None, 23, 9, 9))
    common = convolutional_layer(input, num_filters=512, kernel_size=(3,3))
    common = convolutional_layer(input, num_filters=512, kernel_size=(3, 3))

    # common = residual_layer(common)
    # common = residual_layer(common)
    # common = residual_layer(common)

    vhead = value_head(common)
    phead = policy_head(common)

    losses = {'Policy_Output': 'categorical_crossentropy',
              'Value_Output': 'mean_squared_error'}

    loss_weights = {
        'Policy_Output' : 1,
        'Value_Output': 1
    }

    model = Model(inputs=input, outputs=[vhead, phead])
    model.compile(loss=losses, loss_weights=loss_weights,
                  optimizer='adam')
    return model

def gen_model_small() -> Model:
    input = Input(batch_shape=(None, 5, 9, 9))
    common = convolutional_layer(input, num_filters=512, kernel_size=(3, 3))
    vhead = value_head(common)
    phead = policy_head(common)
    losses = {'Policy_Output': 'categorical_crossentropy',
              'Value_Output': 'mean_squared_error'}

    loss_weights = {
        'Policy_Output': 1,
        'Value_Output': 1
    }

    model = Model(inputs=input, outputs=[vhead, phead])
    model.compile(loss=losses, loss_weights=loss_weights,
                  optimizer='adam')
    return model


VERSIONS={
    1: gen_model,
    2: gen_model_V2,
    3: gen_model_small
}