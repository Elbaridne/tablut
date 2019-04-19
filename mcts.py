memory = {}

from uuid import uuid4
from tablut import Tafl
from math import sqrt, log, exp
from random import random



class Node():
    def __init__(self, state, prior = None):
        self.id = uuid4()
        self.parents = {}
        self.children = {} # {action : Node}
        self.state = state # Tafl 
        self.visit_count = 0
        self.turnplayer = 0
        self.value_sum = 0
        self.prior = prior or 0
    
    def expand(self):
        for piece, destinations in self.state.availableMoves.items():
            for dest in destinations:
                action = (piece, dest)
                n = Node(self.prior, self.state.step(action))
                n.parents[action] = self
                self.children[action] = n
        
    def expanded(self):
        return len(self.children) > 0
    
   
    def value(self):
        return 0 if self.visit_count == 0 else self.value_sum/self.visit_count
    

class MCTS():
    def __init__(self):
        self.nodes = dict()
        self.num_simulations = 200

    def select_action(self, game : Tafl, root : Node):
        visits = [(child.visit_count, action)
                  for action, child in root.children.items()]
        _, action = max(visits)
        return action

    def run(self, game : Tafl):
        root = Node(game)
        
        for _ in range(self.num_simulations):
            game_c = game.clone()
            node = root
            path = [node]

            while node.expanded():
                action, node = self.select_child(node)
                game_c.step(action)
                path.append(node)
            
            def evaluate(node, game_c, e):
                return random()

            value = evaluate(node, game_c, None)

            self.backward_prop(path, value, game_c.currentPlayer)
        return self.select_action(game, root), root
    

    def ucb_score(self, parent : Node, child : Node):
        return child.value() + sqrt(log(parent.visit_count)/child.visit_count)

    def select_child(self, node : Node):
        _, action, child = max((self.ucb_score(node, child), action, child)
                            for action, child in node.children.items())
        return action, child

    def backward_prop(self, path : list, value, player):
        for node in path:
            if node.turnplayer == player:
                node.value_sum += value
            else: node.value_sum += (1-value)
            node.visit_count += 1
    


def evaluate(node : Node, game : Tafl, network):
    value_nn, policy_nn = network.inference(game.state)
    node.turnplayer = game.currentPlayer
    policy = {a : exp(policy_nn[a]) for a in game.availableMoves}
    policy_sum = sum(policy.values())
    for action, p in policy.items():
        node.children[action] = Node(p / policy_sum)
    return value

def play(network):
    game = Tafl()
    mcts = MCTS()
    while not game.done and len(game.moveHistory) < 400:
        action, root = mcts.run(game)
        game.step(action)
        game.store_search(root)
    return game