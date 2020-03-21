from multiprocessing import Pool
from utils import timeit
from tablut import Tafl
from random import choice, shuffle
import numpy as np
from graphviz import Digraph
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
            yield
            

    def ucb(self, hijo):
        if hijo.visitas == 0:
            return inf
        else:
            return (hijo.wins/hijo.visitas) + sqrt(2) * sqrt(log(self.visitas)/hijo.visitas)
    
    #@timeit
    def max_ucb(self):
        assert len(self.hijos) != 0, "Los hijos no deberian de ser 0..."
        self.hijos.items()
        a, hijo = max(self.hijos.items(), key= lambda pair_a_hijo: self.ucb(pair_a_hijo[1]))
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


def backpropagate(node : Node, root_id, ganador, player):
    while node.padre != None:
        node.visitas += 1
        node.wins += 1 if ganador == player else 0
        node = node.padre
    node.visitas += 1
    node.wins += 1 if ganador == player else 0
    node.wins -= 1 if ganador == -player else 0


def run_mcts(root, simulations):
    player = root.tafl.currentPlayer
    
   
    # Seleccion
    for _ in range(simulations):
        node = root
        while len(node.hijos) > 0 and len(node.acciones_restantes)==0:
            _, node = node.max_ucb()
        _tafl_clone = node.tafl.clone()
            
        if node.visitas == 0:
            ganador = rollout(_tafl_clone)
            backpropagate(node, root.id, ganador, player)

        else:
            h,_=node.expand_random()    
            ganador = rollout(_tafl_clone)
            backpropagate(h, root.id, ganador, player)
    
    a, _ = root.max_ucb()
    return a


def play():
    node = Node()
    while not node.terminal():
        if node.tafl.currentPlayer == 1:
            a = run_mcts(node , 50)
        else: 
            a = choice(node.tafl._mask())
        
        print(node.tafl)

    #Jugador Aleatorio
    #Jugador MCTS




def draw_tree(root, comment=''):
    dot = Digraph(comment=comment)
    
    def create_nodes(dot, nodo):
        dot.node(f'{nodo.visitas} {nodo.wins} {len(nodo.hijos)}', shape='box')
        
        for hijo in nodo.hijos.values():
            create_nodes(dot, hijo)

        try:
            for hijo in nodo.hijos.values():
                dot.edge(f'{nodo.id}', f'{hijo.id}')
        except Exception as e:
            pass



    create_nodes(dot, root)
    dot.render(f'test-output/nvm.gv', view=True)