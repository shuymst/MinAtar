import torch
import numpy as np
from buffer import ReplayBuffer
from model import QNetwork

class DQN(object):
    def __init__(self, env):
        self.env = env
        in_channel_num = env.state_shape()[2]
        action_num = env.num_actions()
        self.model = QNetwork(in_channel_num, action_num)
        self.buffer = ReplayBuffer(10000)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def select_action(self, state):
        rand = np.random.random()
        if rand < 0.1:
            action = torch.randint(0, 6)
        else:
            with torch.no_grad():
                action_values= self.model(state)
        
        return action



    def _make_input(self, obs):
        return torch.FloatTensor(obs).permute(2, 0, 1).unsqueeze(1)

    def train(self):
        
        done = False
        while not done:
            state = self._make_input(self.env.state())
            action = self.select_action(state)
            reward, done = self.env.act(action)
            self.buffer.add(state, action, reward, done)
        
        self.env.reset()

