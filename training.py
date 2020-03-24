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
                 num_matches, batch_size,
                 temp_threshold = 30):
        self.p1 = p1
        self.p2 = p2 if p2 is not None else p1
        self.batch_size = batch_size
        self.num_matches = num_matches
        self.env = env
        self.temp_threshold = temp_threshold


    def __len__(self):
        return self.num_matches // self.batch_size

    def __getitem__(self, idx):
        #if tengo suficientes training examples para devolver en el batch
        #       los devuelvo desde cache
        #else:
        # juego X partidas y genero datos

        env = self.env
        output: List[Tuple[int, List[Replay]]] = list() ## TODO Hacer concurrent stack :P
        ## TODO Paralelizar este bloque :D
        for _ in range(self.batch_size):
            env.reset()
            history: List[Replay] = list()
            temp = 1
            self.p1.set_state(env)
            self.p2.set_state(env)

            while True:
                if env.turn > self.temp_threshold:
                    temp = 0
                if env.currentPlayer == self.p1.side:
                    curr = self.p1
                else:
                    curr = self.p2

                if curr.bot:
                    p, a= curr.inference(temp)
                    history.append(Replay(env, p, a))
                    curr.act()
                else:
                    raise NotImplementedError

                if env.done:
                    output.append((env.winner, history))
                    break
            pass

        x = list()
        y = list()

        for gameReplay in output:
            winner = gameReplay[0]
            replayData = gameReplay[1]
            for turn in replayData:
                x.append(NNInputs.from_Tafl(turn.env).to_neural_input())
                y.append([turn.env.currentPlayer * turn.winner, np.array(turn.p).reshape((1, 1296))])


        return x, y

class TrainNeuralNet:
    def __init__(self, net: Model):
        self.net = net
        self.mcts = MCTS(net)
        self.env = Tafl()
        self.trainExamples = []

    def episode(self):
        trainExamples = []
        temp = 1
        env = self.env
        env.reset()
        step = 0
        while True:
            step += 1
            if step > 50:
                temp = 0

            policy = self.mcts.action_probability(env, temp)
            # print(np.argmax(policy))
            action = np.argmax(policy)
            trainExamples.append([env, policy, env.currentPlayer, None])
            env = env.cl_step(action)
            if env.done:
                return [(st, po, pl, pl * env.winner) for st, po, pl, re in trainExamples]

    def train_gen(self):
        pass

    def train(self):
        print('Ejecutando episodio')
        trainExamples = self.episode()
        print('Numero de ejemplos ', len(trainExamples))
        self.net.fit
        for state, policy, player, reward in trainExamples:
            print('Fitting...')
            y = []
            y.append(np.array(reward).reshape((1, 1)))
            y.append(np.array(policy).reshape((1, 1296)))

            self.net.fit(x=NNInputs.from_Tafl(state).to_neural_input(),
                         y=y)
