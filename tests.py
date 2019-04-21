import unittest
from tablut import Tafl
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
        


if __name__ == '__main__':
    unittest.main()