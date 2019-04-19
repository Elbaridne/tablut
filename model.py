from keras import Sequential
from keras.layers import Dense, Conv2D, Flatten
from tablut import Tafl, SPACE_ACTION
import numpy as np

def model():
    model = Sequential()
    model.add(Conv2D(162, kernel_size=3, strides=(2,2), activation='relu', input_shape=(9,9,15)))
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
    reshaped = np.reshape(env.state, (1,9,9,1))
    for time in range(500):
        prediction = model.predict(reshaped)
        len(prediction)


