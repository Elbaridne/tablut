from dataclasses import dataclass
from itertools import chain

import numpy as np

from tablut import Tafl, SIZE, ACTION_SPACE
from typing import *


@dataclass
class NNInputs:
    king: np.array((SIZE, SIZE))
    structures: np.array((SIZE, SIZE))
    muscovite: np.array((SIZE, SIZE))
    swedish: np.array((SIZE, SIZE))
    player: int
    moves: np.array

    @staticmethod
    def from_Tafl(tafl: Tafl):
        return NNInputs(
            NNInputs.pieces_to_arr(tafl._pieces((33,))),
            NNInputs.pieces_to_arr(chain(tafl._pieces((22,)), tafl._pieces((55,)))),
            NNInputs.pieces_to_arr(tafl._pieces((44,))),
            NNInputs.pieces_to_arr(tafl._pieces((11,))),
            tafl.currentPlayer,
            tafl.matrix_moves())

    @staticmethod
    def pieces_to_arr(arr: Union[list, chain]):
        out = np.zeros((9,9))
        for x,y in arr:
           out[x,y] = 1
        return out

    @staticmethod
    def apply_mask(tafl: Tafl, arr: Union[list, chain]) -> np.array:
        actions = np.array(tafl.mask)
        mask = np.ones((1, 1296))
        return np.where(actions in mask, mask, 0)

    def pad_moves(self):
        conc_n = 16 - self.moves.shape[0]
        return np.concatenate([self.moves, np.zeros((conc_n, 9, 9))])

    def to_neural_input(self):
        if self.player == 1:
            return np.concatenate((np.array([self.king,
                             self.structures,
                             self.muscovite,
                             self.swedish,
                             np.ones((SIZE, SIZE)),
                             ]), self.pad_moves()))
        else:
            return np.concatenate((np.array([self.king,
                             self.structures,
                             self.muscovite,
                             self.swedish,
                             np.zeros((SIZE, SIZE)),
                             ]), self.pad_moves()))