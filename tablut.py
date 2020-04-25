# coding=utf-8
from collections import namedtuple
from dataclasses import dataclass

from utils import rngx, rngy, timeit
import numpy as np
from itertools import takewhile, chain, cycle
from functools import reduce
from hashlib import sha1
from pprint import pprint as _print
from random import choice
from copy import deepcopy
from typing import *
import json, base64

NAME = "Tablut"
SIZE = 9
MAX_MOVES = 100
DIRECTIONS = {'down': (1, 0), 'up': (-1, 0), 'right': (0, 1), 'left': (0, -1)}
TEAM = {1: 'Muscovites', -1: 'Swedish', 0: 'None'}

# https://stevenloria.com/lazy-properties/
def lazy_property(fn):
    '''Decorator that makes a property lazy-evaluated.
    '''
    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)
    return _lazy_property


def action_spaces():
    """Returns all possible actions in the board as a pair of positions
        (from_pos , to_pos) where pos is (x,y)"""
    all_actions = list()
    for i in range(SIZE):
        for j in range(SIZE):
            ch = list(chain(rngx(i + 1, j, SIZE, 1),
                            rngy(i, j - 1, -1, -1),
                            rngy(i, j + 1, SIZE, 1),
                            rngx(i - 1, j, -1, -1)
                            ))
            all_actions.extend(map(lambda x: (i, j, *x), ch))
    return all_actions


ACTION_SPACE = {v: k for k, v in enumerate(action_spaces())}
SPACE_ACTION = {k: v for k, v in enumerate(action_spaces())}
Piece = namedtuple('Piece', ['x', 'y'])



class Tafl:
    """Engine for the Hnefatafl nordic games"""
    CASTLES = [(0, 0), (0, SIZE - 1), (SIZE - 1, 0), (SIZE - 1, SIZE - 1)]
    def __init__(self, board=None, currentPlayer=None, winner=None, done=None, turn=0, fromjson=None):
        if fromjson:
            fromjson = json.loads(fromjson)
            board = np.array(fromjson['outstr'])
            stats = fromjson['stats']
            winner, currentPlayer, done, turn = stats['w'], stats['p'], stats['d'], stats['t']

        self.board = board if board is not None else self._new_board()
        self.currentPlayer = currentPlayer or 1
        self.winner = winner or 0
        self.action_space = ACTION_SPACE
        self.space_action = SPACE_ACTION
        self.turn = turn
        self.done = done or self.turn >= MAX_MOVES


    @property
    def mask(self):
        return self._mask()

    @property
    def argmask(self):
        return self._argmask()

    def __hash__(self):
        return int(sha1(self.board).hexdigest(), 16) + self.currentPlayer

    def __str__(self):
        out = '\ A  B  C  D  E  F  G  H  I\n'
        for idx, x in enumerate(self.board):
            out += str(idx) + ' '
            for y in x:
                out += str(y).zfill(2) + ' ' if y != 0 else '   '
            out += "|\n"
        return out

    def __repr__(self):
        return f'''We are playing {NAME}\nCurrent player is {TEAM[self.currentPlayer]}, Turn: {self.turn}\nFinished: {self.done}, Winner: {TEAM[self.winner]}\n {self.board}'''

    @staticmethod
    def reset():
        return Tafl()

    def json(self):
        tojson = {
            'outstr': self.board.tolist(),
            'stats': {
                'p': self.currentPlayer,
                't': self.turn,
                'd': self.done,
                'w': self.winner
            }
        }
        print(tojson['outstr'])
        return json.dumps(tojson)

    def in_step(self, index_action):
        from_p, to_p = self.action_decode(index_action)
        self.board, self.done, self.winner = self._check_move(from_p, to_p)
        self.currentPlayer = -self.currentPlayer
        self.turn += 1
        if self.turn >= MAX_MOVES:
            self.done = True
        return True

    def cl_step(self, index_action):
        from_p, to_p = self.action_decode(index_action)
        # Returns the updated board
        newboard, is_done, winner = self._check_move(from_p, to_p)
        # Another Talf object is created instead of updating the current one
        return Tafl(newboard, -self.currentPlayer, winner, is_done, self.turn + 1,)

    def clone(self):
        return Tafl(self.board, self.currentPlayer, self.winner, self.done, self.turn)

    def _new_board(self):
        """
        Return a NxN numpy matrix, representation of the game board
        with the tablut pieces in right place.
        22 - Castle, 44 - Muscovites, 11 - Swedish, 33 - King 0 - Free Space
        """
        # TODO Use size to get other Half configurations
        muscov = [(0, 3), (0, 4), (0, 5), (1, 4)]
        swedish = [(2, 4), (3, 4)]
        castle = [[0, 0], [0, SIZE - 1], [SIZE - 1, 0], [SIZE - 1, SIZE - 1]]

        board = np.zeros((SIZE, SIZE), dtype=np.uint8)

        # Placing the castles at the corners
        for sx, sy in castle:
            board[sx, sy] = 22

            # Swedish and Muscovites
        for _ in range(4):
            for mosco in muscov:
                board[mosco] = 44
            for sueco in swedish:
                board[sueco] = 11
            board = np.rot90(board)

        # King
        board[(4, 4)] = 33

        return board

    def _pieces(self, team):
        """
            Returns a list of tuples [(x,y), ...] representing
            where the specified pieces are placed.
        """
        x, y = np.where(self.board == team)
        return zip(x, y)

    def _collisions(self, piece: tuple, squares) -> Iterator:
        """
            Appends moves to the list while there are no collisions with other pieces.
        """
        x, y = piece

        # Appends position while the square is free in all cardinals
        rt = (el for el in takewhile(lambda ps: self.board[ps] in squares, rngy(x, y + 1, 9, 1)))
        lf = (el for el in takewhile(lambda ps: self.board[ps] in squares, rngy(x, y - 1, -1, -1)))
        up = (el for el in takewhile(lambda ps: self.board[ps] in squares, rngx(x + 1, y, 9, 1)))
        do = (el for el in takewhile(lambda ps: self.board[ps] in squares, rngx(x - 1, y, -1, -1)))
        return chain(rt, lf, up, do)

    def _available_moves(self, piece: Piece) -> Optional[Iterator]:
        """
            Given a position of a piece as (x,y), return a lisf of
            the available moves in all cardinal directions.
        """
        # Wrong piece
        if self.board[piece] in (0, 22):
            return None

            # Swedish or Muscovite pawn
        if self.board[piece] in (44, 11):
            return self._collisions(piece, (0,))

        # King: he can also move to 22 to win
        else:
            return self._collisions(piece, (0, 22))

    def _all_moves_team(self) -> dict:
        """
            Returns a dict {piece : available_moves} for all pieces of the current player
            where piece = (x,y) and available_moves [(x1,y1)..]
        """
        if self.currentPlayer == 1:
            piezas = self._pieces((44))
        else:
            piezas = chain(self._pieces(11), self._pieces(33))

        moves = {pieza: list(self._available_moves(pieza)) for pieza in piezas}
        # Remove pieces with no movements available
        return {k: v for k, v in moves.items() if len(v) > 0}

    def matrix_moves(self):
        x = self._all_moves_team()
        moves_matrices = list()
        for k, v in x.items():
            arr = np.zeros((SIZE, SIZE))
            arr[k] = -1
            for m in v:
                arr[m] = 1
            moves_matrices.append(arr)
        return np.array(moves_matrices)

    def _mask(self) -> list:
        """
            Returns the move mask of all available pieces
        """
        move_dict = self._all_moves_team()
        valid_moves = [self.action_enc(*p, *move) for p, m in move_dict.items() for move in m]
        return valid_moves

    def _rotate_mask_90(self):
        for piece, moves in self._all_moves_team().items():
            for move in moves:
                x = np.zeros((9,9))
                x[piece] = -1
                x[move] = 3
                x = np.rot90(x)
                i = np.where(x == -1)
                f = np.where(x == 3)
                print(piece, move, i, f)



    def _raichi_tuichi(self) -> int:
        """0 - No winning moves, 1/2 winning moves"""
        pos_rey = tuple(*self._pieces((33,)))
        return sum(1 for coll in self._collisions(pos_rey, (0, 22)) if coll in self.CASTLES)

    def _argmask(self):
        move_dict = self._all_moves_team()
        mask = np.zeros((len(ACTION_SPACE)))
        for p, m in move_dict.items():
            for move in m:
                mask[self.action_enc(*p, *move)] = 1
        return mask


    def _check_move(self, from_p, to_p) -> np.array:
        """
            Handles movement logic, places the throne in the board if the king moves
            away from it and calls the infiltrate function to remove slayed pieces
            return : board, is_done, winner
        """
        piece = self.board[from_p]
        copyboard = np.copy(self.board)
        # King
        if piece == 33:
            if copyboard[to_p] == 22:
                # Moving to a Castle, Swedish have won!
                copyboard[from_p], copyboard[to_p] = 0, 33
                return copyboard, True, -1
            elif from_p == (4, 4):
                # King moving away from the throne, we need to place it in the board   
                copyboard[from_p], copyboard[to_p] = 55, 33
            else:
                # Rest of movements
                copyboard[to_p], copyboard[from_p] = copyboard[from_p], copyboard[to_p]

        # Swedish or Muscovites pawns
        else:
            copyboard[to_p], copyboard[from_p] = copyboard[from_p], copyboard[to_p]

            # To apply capture logic we need to check surrounding squares at distance 1 and 2
            dis1, dis2 = self._proximity(copyboard, to_p, 1), self._proximity(copyboard, to_p, 2)
            captured = self._infiltrate(copyboard, piece, dis1, dis2)
            # Remove captured pieces from the board
            if captured is not None:
                for capped in captured:
                    slay = tuple(map(lambda x: sum(x), zip(to_p, DIRECTIONS[capped])))
                    if copyboard[slay] == 33:
                        return copyboard, True, 1
                    copyboard[slay] = 0

        # Game continues
        return copyboard, False, 0

    def _infiltrate(self, board, piece, dis1, dis2):
        """
            Applies the game rules to check for captures.
            Example; If a swedish pawn new_pos has one or more muscovite pawns in the surroundings
            we check if each of those musco pawns have a wall, special square or other swedish pawn
            behind them. To capture the king it has to be trapped in all cardinal directions.
        """
        out = list()
        for direction, square in dis1.items():
            if square != piece and square not in (0, 22, 55):
                # If a surrounding piece is an enemy, check piece behind it
                if dis2.get(direction, None) in (piece, 22, 55, None):
                    if square == 33 and piece == 11:
                        continue
                    if square == 33:
                        if self._king_trapped(board): out.append(direction)
                    else:
                        out.append(direction)
        return out if len(out) > 0 else None

    def _king_trapped(self, board):
        """
            Returns False if at least one surrounding square of the king is an ally or free space
        """
        king_pos = list(self._pieces((33,)))
        surrounding_pos = self._proximity(board, *king_pos, n=1)
        surr_pieces = surrounding_pos.values()
        free_or_ally = list(filter(lambda x: x == 0 or x == 11, surr_pieces))
        return True if len(free_or_ally) == 0 else False

    def _proximity(self, board, piece, n):
        """
            Returns a Dict with all cardinal positions and the piece
            at n distance of that direction
        """
        x, y = piece
        directions = dict()
        if x + n <= 8:
            directions['down'] = board[x + n, y]
        if x - n >= 0:
            directions['up'] = board[x - n, y]
        if y + n <= 8:
            directions['right'] = board[x, y + n]
        if y - n >= 0:
            directions['left'] = board[x, y - n]
        return directions

    # Actions
    @staticmethod
    def action_decode(index) -> Tuple:
        """Index to action tuple"""
        x1, y1, x2, y2 = SPACE_ACTION[index]
        return (x1, y1), (x2, y2)

    @staticmethod
    def action_dec(index):
        return list(SPACE_ACTION[index])

    @staticmethod
    def action_enc(x1, y1, x2, y2):
        return ACTION_SPACE[(x1, y1, x2, y2)]

    @staticmethod
    def action_encode(action):
        """Single action index"""

        return ACTION_SPACE[action]


