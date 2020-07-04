import unittest
from tablut import Tafl
from az_mcts import MCTS
from model import gen_model
from nn_input import NNInputs
from utils import timeit
from time import time
from random import choice
from training import TrainNeuralNet

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

    
    def _test_prediction(self):
        game = Tafl()
        neuralnet = gen_model()
        mcts = MCTS(neuralnet)
        mcts.perform_search(game)


    def _test_actionprob(self):
        game = Tafl()
        neuralnet = gen_model()
        mcts = MCTS(neuralnet)
        mcts.action_probability(game)


    def _test_episode(self):
        neuralnet = gen_model()
        train = TrainNeuralNet(neuralnet)
        output = train.episode()
        for out in output:
            for e in out[1]:
                if e > 0:
                    print(e)

    def _test_train(self):
        neuralnet = gen_model()
        train = TrainNeuralNet(neuralnet)
        train.train()

if __name__ == '__main__':
    unittest.main()