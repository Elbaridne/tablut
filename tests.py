import unittest
from tablut import Tafl
from tree import Node, run_mcts
from utils import timeit
from time import time
from random import choice

def retry(n_times):
    def function(function):
        def wrapper(*args, **kwargs):
            results = list()
            for _ in range(n_times):
                out = function(*args, **kwargs)
                results.append(out)
            print(f'{sum(results)/n_times} in {n_times} times')
        return wrapper
    return function


class TestTafl(unittest.TestCase):

    @retry(100)
    def test_rollout_clone(self):
        before = time()
        game = Tafl()
        while not game.done:
            game = game.cl_step(choice(game.mask))
        return time() - before
        

    @retry(100)
    def test_rollout_inobj(self):
        before = time()
        game = Tafl()
        while not game.done:
            game.in_step(choice(game.mask))
        return time() - before
    
    
    def test_run_100_simulations(self):
        game = Tafl()
        a = Node(game)
        run_mcts(a, 100)
        print(a)
        
    def test_run_200_simulations(self):
        game = Tafl()
        a = Node(game)
        run_mcts(a, 200)

    def test_run_1000_simulations(self):
        game = Tafl()
        a = Node(game)
        run_mcts(a, 1000)
    
    def test_run_rollout(self):
        game = Tafl()
        a = Node(game)





if __name__ == '__main__':
    unittest.main()