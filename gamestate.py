from tablut import Tafl
import numpy as np
from collections import deque

class GameState:
    def __init__(self, black = None, white = None, turnplayer, *args, **kwargs):
        self.black = deque(maxlen=7)
        self.white = deque(maxlen=7)
        self.player = turnplayer
        self.turn = np.full((9,9), turnplayer)
    
    def step(tafl : Tafl, action):
        s = GameState(self.black, self.white, -self.player)
        board = tafl.cl_step(action).board
        black = np.array(board)
        black[black == 111] = 0
        black[black == 333] = 0
        s.black.append(black)

        white = np.array(board)
        white[white == -111] = 0
        s.white.append(white)
        
        return s
    



    


    