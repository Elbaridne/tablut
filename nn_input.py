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
    def pieces_to_arr(arr: Union[list, chain]) -> np.array:
        out = np.zeros((9, 9))
        for x, y in arr:
            out[x, y] = 1
        return out

    @staticmethod
    def apply_mask(tafl: Tafl, arr: Union[list, chain]) -> np.array:
        for idx in range(len(arr)):
            if tafl.argmask[idx] != 1:
                arr[idx] = 0
        return arr

    @staticmethod
    def parse_prediction(tafl: Tafl, prediction: np.array) -> Tuple[np.array]:
        value, policy = prediction
        value = value.reshape((1))
        policy = policy.reshape((1296))
        masked_policy = NNInputs.apply_mask(tafl, policy)
        return value, masked_policy

    def pad_moves(self) -> np.array:
        conc_n = 16 - self.moves.shape[0]
        return np.concatenate([self.moves, np.zeros((conc_n, 9, 9))])

    def to_neural_input(self):

        if self.player == 1:
            arr_player = np.ones((SIZE, SIZE))
        else:
            arr_player = np.zeros((SIZE, SIZE))

        # output shape -> (1, 21, 9, 9)
        arr = np.concatenate((np.array([self.king,
                                         self.structures,
                                         self.muscovite,
                                         self.swedish,
                                         arr_player,
                                         ]), self.pad_moves()))
        return arr[np.newaxis, :]


