from multiprocessing import Pool
from utils import timeit
from tablut import Tafl, TEAM
from random import choice, shuffle
import numpy as np

from math import log, sqrt, inf
from uuid import uuid4
import time
# {hash(state) : Node}
states = dict()

class Node:
    def __init__(self, estado = None, padre = None):
        self.id = uuid4()
        self.padre = padre
        self.tafl = estado or Tafl()
        self.acciones_restantes = self.tafl.mask
        self.hijos = dict()
        self.visitas = 0
        self.wins = 0
        self.turn = self.tafl.turn
        self.value = self.tafl.winner

    def go_to_child(self, a):
        hijo = self.hijos[a]
        del self.hijos
        del self
        return hijo
    
    def expand_random(self):
        try:
            a = self.acciones_restantes.pop()
            node = Node(self.tafl.cl_step(a), self)
            self.hijos[a] = node
            return self.hijos[a], a
        except Exception as e:
            print('Why am I HERE')

    #@timeit
    def expand_all(self):
        while len(self.acciones_restantes) > 0:
            a = self.acciones_restantes.pop()
            node = Node(self.tafl.cl_step(a), self)
            self.hijos[a] = node

            

    def ucb(self, hijo, temp):
        if hijo.visitas == 0:
            return inf
        else:
            if temp is None:
                return (hijo.wins/hijo.visitas) + sqrt(2) * sqrt(log(self.visitas)/hijo.visitas)
            else:
                return (hijo.wins / hijo.visitas) + temp * sqrt(log(self.visitas) / hijo.visitas)
    
    #@timeit
    def max_ucb(self, temp=None):
        assert len(self.hijos) != 0, "Los hijos no deberian de ser 0..."
        self.hijos.items()
        a, hijo = max(self.hijos.items(), key= lambda pair_a_hijo: self.ucb(pair_a_hijo[1], temp))
        return a, hijo


    def terminal(self):
        return self.tafl.done

    def __repr__(self):
        return f"Node<a_resta: {len(self.acciones_restantes)} n_hijos:{len(self.hijos)}, visitas:{self.visitas}, wins: {self.wins}>"


#Simulacion

def rollout(tafl : Tafl):
    while not tafl.done:
        tafl = tafl.cl_step(choice(tafl.mask))
    return tafl.winner


def backpropagate(node : Node, ganador):
    while node.padre != None:
        node.visitas += 1
        if ganador != 0:
            if ganador == node.tafl.currentPlayer:
                node.wins += 1
            else:
                node.wins -= 1

        node = node.padre
        ganador = -ganador

    # Llegamos al Padre
    node.visitas += 1
    if ganador != 0:
        if ganador == node.tafl.currentPlayer:
            node.wins += 1
        else:
            node.wins -= 1



def run_mcts(root, simulations, temp=None):
    player = root.tafl.currentPlayer
    
   
    # Seleccion
    for _ in range(simulations):
        node = root
        while node.visitas != 0:
            node.expand_all()
            _, node = node.max_ucb(temp)

        _tafl_clone = node.tafl.clone()
        w = rollout(_tafl_clone)
        backpropagate(node, w)
            

    
    a, _ = root.max_ucb()
    print(F"Veo {root.wins} WINS en el futuro del jugador {TEAM[player]}")
    p = root.go_to_child(a)

    return a, p


def play():
    node = Node()
    while not node.terminal():
        if node.tafl.currentPlayer == 1:
            a = run_mcts(node , 50)
        else: 
            a = choice(node.tafl._mask())
        node.tafl.in_step(a)
        print(node.tafl)


    #Jugador Aleatorio
    #Jugador MCTS


if __name__ == '__main__':
    play()
