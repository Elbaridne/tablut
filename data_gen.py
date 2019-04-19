from tablut import Tafl
import numpy as np
from random import choice


board = lambda : Tafl()._new_board()
winner = lambda board : Tafl(board)._king_trapped(board)


def gen_random_wins(num):
    suecos100 = list()
    moscos100 = list()
    while len(moscos100) <= num:
        env = Tafl()
        while env.winner != -1 and not env.done:
            env.step(choice(env.mask))
        if env.winner == 1:
            moscos100.append(env.state)
        if env.winner == -1:
            #suecos100.append(env.state)
            pass
        env.reset()
    return suecos100, moscos100

def augment_data(array):
    f2 = np.flip(array, 0)
    sols = (array, f2)
    outs = [np.rot90(array) for array in sols for _ in range(4)]
    return outs



load = lambda file  : np.load(file) # load
save = lambda x, name : np.save(f'{name}.npy', x) # save