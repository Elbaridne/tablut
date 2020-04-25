from typing import Union, Optional, List, Tuple
from pprint import pprint
from tensorflow.keras import Model
from tensorflow.keras.utils import Sequence

from az_mcts import MCTS
from tablut import Tafl, SPACE_ACTION, ACTION_SPACE
from model import VERSIONS
from random import choice
import numpy as np
from nn_input import NNInputs
from collections import deque, namedtuple
import pickle, os
import datetime
import random

Replay = namedtuple('Replay', ['env', 'p', 'a'])


class Supervisor:
    def __init__(self, num_sim, batch_size, weights_path=None, model_ver=2):
        model = VERSIONS[model_ver]()
        if weights_path is not None:
            if os.path.exists(weights_path):
                print('Weights found')
                model.load_weights(weights_path)
        else:
            weights_path = 'modelv1.checkpoint.h5'

        self.model = model
        self.weights_path = weights_path
        self.train_helper = TrainNeuralNet(model, num_sim=num_sim, batch_size=batch_size)

    def save(self):
        if self.model.save_weights(self.weights_path):
            return True
        return False

    def train(self, epochs, replay):
        for i in range(epochs):
            self.train_helper.train_gen(replay=replay)


class Agent():
    def __init__(self, state: Tafl, side, bot=True, model=None, num_sim=None):
        self.state = state
        self.bot = bot
        self.side = side
        if bot:
            assert model is not None
            self.model = model
            self.mcts = MCTS(self.model, num_sim=num_sim)

    def set_state(self, state: Tafl):
        self.state = state

    def act(self, a) -> Tafl:
        return self.state.in_step(a)

    def inference(self, temp):
        policy = self.mcts.action_probability(self.state, temp)
        action = np.random.choice(np.arange(len(policy)),
                                  p=policy
                                  )
        print(action)
        # print(np.argmax(policy))
        # trainExamples.append([env, policy, env.currentPlayer, None])
        return policy, action

    def player_input(self):
        pass


class Arena(Sequence):

    def __init__(self, env: Tafl,
                 p1: Agent,
                 p2: Optional[Agent],
                 batch_size, steps=100,
                 temp_threshold=30, replay=False):
        self.p1 = p1
        self.p2 = p2 if p2 is not None else p1
        self.batch_size = batch_size
        self.env = env
        self.temp_threshold = temp_threshold
        self.replay_history = list()
        self.replay = replay

        if self.replay:
            self.load()

    def load(self):
        for repl in os.listdir('replays/'):
            with open('replays/' + repl, 'rb') as f:
                history = pickle.load(f)
                self.replay_history.extend(history)

    def __len__(self):
        return 200

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
                print(SPACE_ACTION[a])
                curr.act(a)
            else:
                raise NotImplementedError

            if env.done:
                for step in history:
                    # Augment and append to Replay History
                    self.augment(step, env.winner)

                return env.winner

    def augment(self, step: Replay, winner):
        first = NNInputs.from_Tafl(step.env)
        first_mask = np.array(step.p).reshape(1296)
        self.replay_history.append((
            first.to_neural_input(),
            [np.array(step.env.currentPlayer * winner),
             first_mask]))

        secnd = first.rot90()
        secnd_mask = NNInputs.rot90_mask(first_mask)
        self.replay_history.append((
            secnd.to_neural_input(),
            [np.array(step.env.currentPlayer * winner),
             secnd_mask]))

        third = secnd.rot90()
        third_mask = NNInputs.rot90_mask(secnd_mask)
        self.replay_history.append((
            third.to_neural_input(),
            [np.array(step.env.currentPlayer * winner),
             third_mask]))

        fourt = third.rot90()
        fourt_mask = NNInputs.rot90_mask(third_mask)
        self.replay_history.append((
            fourt.to_neural_input(),
            [np.array(step.env.currentPlayer * winner),
             fourt_mask]))

    def __getitem__(self, idx):
        while len(self.replay_history) < self.batch_size:
            if self.replay:
                self.load()
            else:
                self.play()

        training_examples = self.replay_history[self.batch_size * idx: self.batch_size * (idx + 1)]
        if not self.replay:
            with open(
                    f'replays/Tafl-t{self.env.turn}-{self.batch_size}-{datetime.datetime.now().strftime("%d_%m %H_%M_%S")}-{random.randint(1, 100000)}.pkl',
                    'wb') as f:
                pickle.dump(training_examples, f)

        # del self.replay_history[0:self.batch_size]
        x = np.array([example[0] for example in training_examples])
        v = list(example[1][0] for example in training_examples)
        p = list(example[1][1] for example in training_examples)
        print(np.array(v).shape)
        y = [np.array(v).reshape(self.batch_size), np.array(p).reshape(self.batch_size, 1296)] # TODO Dando errores
        print(p, y)
        return x, y


class TrainNeuralNet:
    def __init__(self, net: Model, num_sim, batch_size):
        self.net = net
        self.env = Tafl()
        self.num_sim = num_sim
        self.batch_size = batch_size
        self.trainExamples = []

    def train_gen(self, replay=False):
        ag1 = Agent(self.env, 0, True, self.net, num_sim=self.num_sim)
        ar = Arena(self.env, ag1, None, self.batch_size, replay=replay)
        self.net.fit(ar)


if __name__ == '__main__':
    supervisor = Supervisor(batch_size=8, num_sim=2)
    supervisor.train(epochs=10, replay=True)
    supervisor.save()
