# coding=utf-8
import numpy as np
from itertools import takewhile, chain, cycle
from pprint import pprint as _print
from random import choice
import pandas as pd
import time
from copy import deepcopy
import json, base64


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print ('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed

# class Environment:
#     def __init__(self, state=None):
#         self.state = state or Tafl()
#         self.actionSpace = self.state._action_spaces()
#         self.grid_shape = (9,9)
#         self.input_shape = (2,9,9)
#         self.state_size = 9*9*2
#         self.action_size = len(self.actionSpace)
#         self.done = False
#         self.turn = 0
#         self.winner = 0
    
#     def reset(self):
#         self = Tafl()

    
#     def __repr__(self):
#         return f"Turn {self.turn} : " + self.state.__repr__()

#     def clone(self):
#         return Environment(self.state)
        

#     def step(self, action):
#         oldp, newp = self.state.decode_action(action)
#         self.moveHistory.append((self.state.currentPlayer, action))
#         self.state.turn += 1
        
#         # Game Logic after each move
#         copyState, done, winner = self.state._check_move(oldp, newp)
#         self.state.gameState = copyState
#         self.state.currentPlayer = -self.state.currentPlayer
#         self.availableMoves = self.state._all_moves_team()
#         self.done, self.state.done = done, done
#         self.state.winner, self.winner = winner, winner
#         return (copyState, done, winner)


direci = {'down': (1,0), 'up': (-1,0), 'right': (0,1), 'left':(0,-1)}
# Lambda Functions to get ranges in the board

rngx = lambda x,y,lim,st: ((_x, y) for _x in range(x, lim, st))
rngy = lambda x,y,lim,st: ((x, _y) for _y in range(y, lim, st))
equipo = {1:'Muscovites', -1:'Swedish', 0:'None'}
SIZE = 9
MAX_MOVES = 200
NAME = "Tablut"

def action_spaces():
        '''Returns all possible actions in the board
           returns: [((x,y),(x1,y1)), ...]
        '''
        all_actions = list()
        for i in range(SIZE):
            for j in range(SIZE):
                ch =list(chain(rngx(i+1,j,9,1), 
                    rngy(i,j-1,-1,-1), 
                    rngy(i,j+1,9,1), 
                    rngx(i-1,j,-1,-1)
                ))
                all_actions.extend(map(lambda x : (i,j,*x), ch))
        
        return all_actions


ACTION_SPACE = {v:k for k,v in enumerate(action_spaces())}
SPACE_ACTION = {k:v for k,v in enumerate(action_spaces())}




class Tafl:
     
    def __init__(self, state = None, currentPlayer = None, winner = None, done = None, turn = 0, fromjson = None):
        if fromjson:
            fromjson = json.loads(fromjson)
            state = np.array(fromjson['outstr'])
            stats = fromjson['stats']
            winner, currentPlayer, done, turn  = stats['w'], stats['p'], stats['d'], stats['t']

        self.currentPlayer = currentPlayer or 1
        self.state = state if state is not None else self._new_board()
        self.action_space = ACTION_SPACE
        self.space_action = SPACE_ACTION
        #self.legal_actions = self._valid_moves()
        self.turn = turn
        self.done = done or self.turn >= 200 
        self.winner = winner or 0
        self.mask = self._valid_moves()
        
    
    def __repr__(self):
        return f'''We are playing {NAME}\nCurrent player is {equipo[self.currentPlayer]}, Turn: {self.turn}\nFinished: {self.done}, Winner: {equipo[self.winner]}\n {self.state}'''
   
    def reset(self):
        self = Tafl()    
    
    def json(self):
        tojson =  {
            'outstr' : self.state.tolist(),
            'stats' : {
                'p' : self.currentPlayer,
                't' : self.turn,
                'd' : self.done,
                'w' : self.winner
            }
        }
        print(tojson['outstr'])
        return json.dumps(tojson)

        
    @timeit
    def in_step(self, index_action):
        from_p, to_p = self.action_decode(index_action)
        # Returns the updated board
        newstate, is_done, winner = self._check_move(from_p, to_p)
        self.state = newstate
        self.currentPlayer = -self.currentPlayer
        self.turn += 1
        self.winner = winner
        #self.legal_actions = self._valid_moves()
        self.mask = self._valid_moves()
        self.done = is_done or len(self.mask) == 0
        # Another Talf object is created instead of updating the current one
        
    def cl_step(self, index_action):
        from_p, to_p = self.action_decode(index_action)
        # Returns the updated board
        newstate, is_done, winner = self._check_move(from_p, to_p)
        # Another Talf object is created instead of updating the current one
        return Tafl(newstate, -self.currentPlayer, winner, is_done, self.turn+1)
        
    
    def clone(self):
        return Tafl(self.state, self.currentPlayer, self.winner, self.done, self.turn)

    
    def _new_board(self):
        '''
        Return a NxN numpy matrix, representation of the game board
        with the tablut pieces in right place.
        22 - Castle, 44 - Muscovites, 11 - Swedish, 33 - King 0 - Free Space
        '''
        # TODO Use size to get other Half configurations
        muscov = [(0,3),(0,4),(0,5), (1,4)]
        swedish = [(2,4), (3,4)]
        castle = [[0, 0], [0, SIZE-1], [SIZE-1, 0], [SIZE-1, SIZE-1]]
        
        board = np.zeros((SIZE,SIZE), dtype=np.uint8)
        
        # Placing the castles at the corners
        for sx,sy in castle:
            board[sx,sy] = 22 

        # Swedish and Muscovites
        for _ in range(4):
            for mosco in muscov:
                board[mosco] = 44
            for sueco in swedish:
                board[sueco] = 11
            board = np.rot90(board)
        
        # King
        board[(4,4)] = 33

        return board
    
    
    def _pieces(self, team):
        '''
            Returns a list of tuples [(x,y), ...] representing 
            where the specified pieces are placed.
        '''    
        x, y = np.where(self.state == team)
        return zip(x,y)
        
    

    def _collisions(self, piece, squares):
        '''
            Appends moves to the list while there are no collisions with other pieces.
        '''
        x,y = piece

        
        # Appends position while the square is free in all cardinals
        rt= (el for el in takewhile(lambda ps : self.state[ps] in squares, rngy(x,y+1,9,1)))
        lf= (el for el in takewhile(lambda ps : self.state[ps] in squares, rngy(x,y-1,-1,-1)))
        up= (el for el in takewhile(lambda ps : self.state[ps] in squares, rngx(x+1,y,9,1)))
        do= (el for el in takewhile(lambda ps : self.state[ps] in squares, rngx(x-1,y,-1,-1)))
        return chain(rt,lf,up,do)

    
    def _available_moves(self, piece):
        '''
            Given a position of a piece as (x,y), return a lisf of 
            the available moves in all cardinal directions.
        '''
        # Wrong piece
        if self.state[piece] in (0, 22):
            return 
        
        # Swedish or Muscovite pawn
        if self.state[piece] in (44, 11):
            return self._collisions(piece, (0,))
            
        # King: he can also move to 22 to win
        else:
            return self._collisions(piece, (0,22))

    
    def _all_moves_team(self):
        '''
            Returns a dict {piece : available_moves} for all pieces of the current player
            where piece = (x,y) and available_moves [(x1,y1)..]
        '''
        if self.currentPlayer == 1: piezas = self._pieces((44))
        else:                       
            piezas = chain(self._pieces(11), self._pieces(33))

        
        moves = {pieza : self._available_moves(pieza) for pieza in piezas}
        #Remove pieces with no movements available
        return {k:v for k,v in moves.items() if v != []}


    def _valid_moves(self):
        '''
            List of quadruples (x_f, y_f, x_t, y_t) to use as index of legal moves in 
            the action space.
            returns: (0,0,0,1) ... (8,8,8,7) 
        '''

        move_dict = self._all_moves_team()
        # valid_moves = list()
        # for piece, moves in move_dict.items():
        #     valid_moves.extend(map(lambda move: (*piece, *move), moves))
        #print([(p, move) for p, m in move_dict.items() for move in m])
        valid_moves = [self.action_enc(*p, *move) for p, m in move_dict.items() for move in m]
        return valid_moves


    def _check_move(self, from_p, to_p):
        '''
            Handles movement logic, places the throne in the board if the king moves
            away from it and calls the infiltrate function to remove slayed pieces 
            return : state, is_done, winner
        '''
        piece = self.state[from_p]
        copyState = np.copy(self.state)
        # King
        if piece == 33: 
            if copyState[to_p] == 22:
                # Moving to a Castle, Swedish have won!
                copyState[from_p] , copyState[to_p] = 0, 33
                return copyState, True, -1
            elif from_p == (4,4):
                # King moving away from the throne, we need to place it in the board   
                copyState[from_p] , copyState[to_p] = 55, 33
            else:
                # Rest of movements
                copyState[to_p], copyState[from_p] = copyState[from_p], copyState[to_p]
        
        # Swedish or Muscovites pawns
        else:
            copyState[to_p], copyState[from_p] = copyState[from_p], copyState[to_p]
            
            # To apply capture logic we need to check surrounding squares at distance 1 and 2
            dis1, dis2 = self._proximity(copyState, to_p, 1), self._proximity(copyState, to_p, 2)
            captured = self._infiltrate(copyState, piece, dis1, dis2)

            # Remove captured pieces from the board
            if captured is not None:
                for capped in captured:
                    slay = tuple(map(lambda x : sum(x), zip(to_p, direci[capped])))
                    if copyState[slay] == 33:
                        return copyState, True, 1
                    copyState[slay] = 0
                    
        
        # Game continues
        return copyState, False, 0

    
    def _infiltrate(self, state, piece, dis1,dis2):
        '''
            Applies the game rules to check for captures.
            Example; If a swedish pawn new_pos has one or more muscovite pawns in the surroundings
            we check if each of those musco pawns have a wall, special square or other swedish pawn
            behind them. To capture the king it has to be trapped in all cardinal directions. 
        '''
        out = list()
        for direction, square in dis1.items():
            if square != piece and square not in (0,22,44):
                # If a surrounding piece is an enemy, check piece behind it
                if dis2.get(direction, None) in (piece, 22, 44, None):
                    if square == 33 and piece == 11: 
                        continue
                    if square == 33:
                        if self._king_trapped(state): out.append(direction)       
                    else: out.append(direction)
        return out if len(out) > 0 else None

    
    def _king_trapped(self, state):
        '''
            Returns False if at least one surrounding square of the king is an ally or free space 
        '''
        king_pos = list(self._pieces((33,)))
        surrounding_pos = self._proximity(state,*king_pos, n=1)
        surr_pieces = surrounding_pos.values()
        free_or_ally = list(filter(lambda x: x == 0 or x == 11, surr_pieces))
        return True if len(free_or_ally) == 0 else False
        
    
    def _proximity(self, state, piece, n):
        '''
            Returns a Dict with all cardinal positions and the piece 
            at n distance of that direction
        '''
        x,y = piece
        directions = dict()
        if x+n <= 8:
            directions['down'] = state[x+n,y]
        if x-n >= 0:
            directions['up'] = state[x-n,y]
        if y+n <= 8:
            directions['right'] = state[x,y+n]
        if y-n >= 0:
            directions['left'] = state[x,y-n]    
        return directions

    # Actions
     
    def action_decode(self, index):
        '''Index to action tuple'''
        x1,y1,x2,y2 = self.space_action[index]
        return (x1,y1), (x2,y2)
    
    def action_enc(self, x1,y1,x2,y2):
        return self.action_space[(x1,y1,x2,y2)]

    def action_encode(self, action):
        '''Single action index'''

        return self.action_space[action]
        
 
    def _mask(self):    
        '''List of index of legal actions'''  
        return [self.action_encode(a) for a in self.legal_actions]        
        
