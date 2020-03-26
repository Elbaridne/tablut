from typing import Union, Optional, List, Tuple

from tensorflow.keras import Model
from tensorflow.keras.utils import Sequence

from az_mcts import MCTS
from tablut import Tafl
from model import gen_model, load_model, save_weights
from random import choice
import numpy as np
from nn_input import NNInputs
from collections import deque, namedtuple
import pickle, os
import datetime

Replay = namedtuple('Replay', ['env', 'p', 'a'])

class Supervisor:
    def __init__(self, weights_path=None):
        if weights_path is not None:
            model = load_model(weights_path)
        else:
            model = gen_model()
            weights_path = 'checkpoint.h5'
        self.model = model
        self.weights_path = weights_path
        self.train_helper = TrainNeuralNet(model)

    def save(self):
        save_weights(self.model, self.weights_path)

    def train(self, epochs):
        for i in range(epochs):
            self.train_helper.train_gen(True)



class Agent():
    def __init__(self, state: Tafl, side, bot=True, model=None):
        self.state = state
        self.bot = bot
        self.side = side
        if bot:
            assert model is not None
            self.model = model
            self.mcts = MCTS(self.model)

    def set_state(self, state: Tafl):
        self.state = state

    def act(self, a) -> Tafl:
        return self.state.in_step(a)


    def inference(self, temp):
        policy = self.mcts.action_probability(self.state, temp)
        print(policy, 'polisi')
        action = np.argmax(policy)
        # print(np.argmax(policy))
        #trainExamples.append([env, policy, env.currentPlayer, None])
        return policy, action

    def player_input(self):
        pass


class Arena(Sequence):

    def __init__(self, env : Tafl,
                 p1: Agent,
                 p2: Optional[Agent],
                 batch_size, steps = 100,
                 temp_threshold = 30, replay = False):
        self.p1 = p1
        self.p2 = p2 if p2 is not None else p1
        self.batch_size = batch_size
        self.env = env
        self.temp_threshold = temp_threshold
        self.replay_history = list()
        self.replay = replay

    def __len__(self):
        return 100

    def play(self):
        self.env = Tafl.reset()
        env = self.env
        print(env)
        history: List[Replay] = list()
        temp = 1
        self.p1.set_state(env)
        self.p2.set_state(env)

        while True:
            print('A new turn')
            if env.turn > self.temp_threshold:
                temp = 0
            if env.currentPlayer == self.p1.side:
                curr = self.p1
            else:
                curr = self.p2

            if curr.bot:
                print(curr.state)
                p, a = curr.inference(temp)
                history.append(Replay(env, p, a))
                curr.act(a)
            else:
                raise NotImplementedError

            if env.done:
                for step in history:
                    self.replay_history.append(
                        (NNInputs.from_Tafl(step.env).to_neural_input(), [np.array(step.env.currentPlayer * env.winner), np.array(step.p).reshape(1296)]))
                return env.winner

    def __getitem__(self, idx):
        if self.replay:
            for repl in os.listdir('replays/'):
                with open('replays/'+repl, 'rb') as f:
                    history = pickle.load(f)
                    self.replay_history.extend(history)
        else:
            while len(self.replay_history) < self.batch_size:
                self.play()
        training_examples = self.replay_history[0:self.batch_size]
        if not self.replay:
            with open(f'replays/Tafl-t{self.env.turn}-{self.batch_size}-{datetime.datetime.now().strftime("%d_%m %H_%M_%S")}.pkl', 'wb') as f:
                pickle.dump(training_examples, f)

        del self.replay_history[0:self.batch_size]
        x = np.array([example[0] for example in training_examples])
        v = list(example[1][0] for example in training_examples)
        p = list(example[1][1] for example in training_examples)
        y = [np.array(v).reshape(self.batch_size, 1), np.array(p).reshape(self.batch_size, 1296)]
        return x, y

class TrainNeuralNet:
    def __init__(self, net: Model):
        self.net = net
        self.mcts = MCTS(net, num_sim=2)
        self.env = Tafl()
        self.trainExamples = []

    def train_gen(self, replay = False, batch_size = 32):
        ag1 = Agent(self.env, 0, True, self.net)
        ar = Arena(self.env, ag1, None, batch_size, replay=replay)
        self.net.fit(ar)

if __name__ == '__main__':
    model = gen_model()
    supervisor = Supervisor('checkpoint.h5')
    supervisor.train(1)
    supervisor.save()