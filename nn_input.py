from dataclasses import dataclass
from itertools import chain

import numpy as np

from tablut import Tafl, SIZE, ACTION_SPACE, SPACE_ACTION
from typing import *


@dataclass
class NNInputs:
    king: np.array((SIZE, SIZE))
    structures: np.array((SIZE, SIZE))
    muscovite: np.array((SIZE, SIZE))
    swedish: np.array((SIZE, SIZE))
    raichi: np.array((SIZE, SIZE))
    tuichi: np.array((SIZE, SIZE))
    player: int
    moves: np.array

    @staticmethod
    def from_Tafl(tafl: Tafl):
        return NNInputs(
            NNInputs.pieces_to_arr(tafl._pieces((33,))),
            NNInputs.pieces_to_arr(chain(tafl._pieces((22,)), tafl._pieces((55,)))),
            NNInputs.pieces_to_arr(tafl._pieces((44,))),
            NNInputs.pieces_to_arr(tafl._pieces((11,))),
            *NNInputs.raichi_taichi(tafl._raichi_tuichi()),
            tafl.currentPlayer,
            tafl.matrix_moves())

    @staticmethod
    def raichi_taichi(num):
        if num == 2:
            return (np.zeros((9,9)), np.ones((9,9)))
        if num == 1:
            return (np.ones((9,9)), np.zeros((9,9)))
        else:
            return (np.zeros((9,9)), np.zeros((9,9)))

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

    def rot90(self):
        return NNInputs(
            np.rot90(self.king),
            np.rot90(self.structures),
            np.rot90(self.swedish),
            np.rot90(self.muscovite),
            self.raichi,
            self.tuichi,
            self.player,
            np.array(list(map(lambda m : np.rot90(m), (m for m in self.moves))))
        )

    @staticmethod
    def rot90_mask(mask: np.array) -> np.array:
        # Asumimos que se pasa argmask
        rot = np.zeros(shape=(1296,))
        for x in np.nonzero(mask)[0]:
            act = SPACE_ACTION[x]
            i, f = NNInputs.rotate_move(*act)
            rot[Tafl.action_enc(*i, *f)] = 1
        return rot

    @staticmethod
    def symm_mask(self):
        pass

    @staticmethod
    def rotate_move(xi,yi,xf,yf):
        return (SIZE - 1 - yi, xi),(SIZE - 1 - yf, xf)

    @staticmethod
    def x_simmetry_move(pos_i, pos_f):
        xi, yi = pos_i
        xf, yf = pos_f
        return (xi, SIZE - 1 - yi), (xf, SIZE - 1 - xf)

    @staticmethod
    def y_simmetry_move(pos_i, pos_f):
        xi, yi = pos_i
        xf, yf = pos_f
        return (SIZE - xi - 1, yi), (SIZE - 1 - xf, yf)

    def pad_moves(self) -> np.array:
        conc_n = 16 - self.moves.shape[0]
        return np.concatenate([self.moves, np.zeros((conc_n, 9, 9))])

    def to_neural_input(self, add_axis=False):

        if self.player == 1:
            arr_player = np.ones((SIZE, SIZE))
        else:
            arr_player = np.zeros((SIZE, SIZE))

        # output shape -> (1, 23, 9, 9)
        arr = np.concatenate((np.array([self.king,
                                         self.structures,
                                         self.muscovite,
                                         self.swedish,
                                         self.raichi,
                                         self.tuichi,
                                         arr_player,
                                         ]), self.pad_moves()))

        if add_axis:
            return arr[np.newaxis, ]
        return arr
