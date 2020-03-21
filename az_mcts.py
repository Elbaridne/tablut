from tablut import Tafl
from tensorflow.keras import Model
from nn_input import NNInputs
from math import *
from typing import *
import numpy as np

CPUT = 0.5

class MCTS:
    """ Lógica del Árbol de Busqueda Monte Carlo
        adaptado para tomar una red neuronal
    """
    S: Dict[int, Tafl]

    def __init__(self, net: Model):
        self.net = net
        self.S = dict()
        self.Qsa = dict()
        self.visits_state = self.Vs = {}
        self.nn_policy_per_state = self.Ps = {}
        self.visits_action_state = self.Sa = {}

    def action_probability(self, state: Tafl, temp=1):
        for i in range(10):
            self.perform_search(state)
        state_hash = hash(state)
        counts = [self.visits_action_state[(state_hash, a)] if (state_hash, a) in self.visits_action_state else 0 for a in range(len(state.argmask))]
        if temp == 0:
            bestA = np.argmax(counts)
            probs = np.zeros((len(state.argmask)))
            probs[bestA] = 1
            return probs

        counts = np.array([x**(1./temp) for x in counts])
        counts = counts / counts.sum()
        return counts




    def perform_search(self, state: Tafl):

        state_hash = hash(state)

        if state_hash not in self.S:
            self.S[state_hash] = state

        if self.S[state_hash].done:
            return -self.S[state_hash].winner

        if state_hash not in self.Ps:
            nn_input = NNInputs.from_Tafl(state)
            prediction = self.net.predict(nn_input.to_neural_input())
            val, policy = NNInputs.parse_prediction(state, prediction)
            self.Ps[state_hash] = policy
            sumPs = np.sum(self.Ps[state_hash])
            if sumPs != 0:
                self.Ps[state_hash] /= sumPs
            else:
                print('WHY ARE WE HERE, JUST TO SUFFER?')
                for e in self.Ps[state_hash]:
                    if e > 0:
                        print(e)

            self.visits_state[state_hash] = 0
            return -val

        cur_best = -float('inf')
        best_act = -1
        actions = self.Ps[state_hash]

        for a in range(len(state.argmask)):
            if actions[a] > 0:
                if (state_hash, a) in self.Qsa:
                    u = self.Qsa[(state_hash,a)] + \
                        CPUT * actions[a] * \
                        sqrt(self.visits_state[state_hash])/(1+self.visits_action_state[(state_hash, a)])
                else:
                    u = CPUT * actions[a] * \
                        sqrt(self.visits_state[state_hash] + 1e-8)
                if u > cur_best:
                    cur_best = u
                    best_act = a


        next_state = state.cl_step(best_act)
        v = self.perform_search(next_state)

        if (state_hash, best_act) in self.Qsa:
            self.Qsa[(state_hash, best_act)] = (self.visits_action_state[(state_hash, best_act)] * self.Qsa[(state_hash, best_act)] + v) / (self.visits_action_state[(state_hash, best_act)] + 1)
            self.visits_action_state[(state_hash, best_act)] += 1
        else:
            self.Qsa[(state_hash, best_act)] = v
            self.visits_action_state[(state_hash, best_act)] = 1

        self.visits_state[state_hash] += 1

        return -v



