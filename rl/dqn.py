import torch
import torch.nn.functional as F
import numpy as np
from minatar import Environment
from model import QNetwork
from utils import device, get_state
from collections import deque
import random
import math

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000000
GAMMA = 0.99

class DQNAgent:
    def __init__(self, env_size = 128, batch_size = 254):
        self.q_network = QNetwork(4, 6).to(device)
        self.target_q_network = QNetwork(4, 6).to(device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(params=self.q_network.parameters(), lr=1e-3)
        self.envs = [Environment(env_name="breakout", random_seed=np.random.randint(0, 1<<30)) for _ in range(env_size)]
        self.scores = [0 for _ in range(env_size)]
        self.buffer = deque(maxlen=100000)
        self.env_size = env_size
        self.batch_size = batch_size
        self.update_cnt = 0
        self.steps_done = 0

    def get_batch_actions(self):
        batch_states = []
        for i in range(self.env_size):
            batch_states.append(get_state(self.envs[i].state()))
        
        batch_states = torch.tensor(np.array(batch_states), device=device).float()
        with torch.no_grad():
            batch_state_action_values = self.q_network(batch_states)
            batch_actions = torch.max(batch_state_action_values, dim=1).indices
        return batch_actions
    
    def optimize(self):
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []

        samples = random.sample(self.buffer, self.batch_size)
        for state, action ,reward, next_state, done in samples:
            batch_states.append(state)
            batch_actions.append([action])
            batch_rewards.append(reward)
            batch_next_states.append(next_state)
            batch_dones.append(done)
        
        batch_states = torch.tensor(np.array(batch_states), device=device).float()
        batch_state_values = self.q_network(batch_states)
        batch_actions = torch.tensor(batch_actions, device=device)
        batch_state_action_values = batch_state_values.gather(1, batch_actions).squeeze()

        batch_next_states = torch.tensor(np.array(batch_next_states), device=device).float()
        batch_next_state_values = self.target_q_network(batch_next_states)
        batch_next_state_action_values = torch.max(batch_next_state_values, dim=1).values
        batch_dones = torch.tensor(batch_dones, device=device).float()
        batch_rewards = torch.tensor(batch_rewards, device=device).float()
        # assert(batch_next_state_action_values.shape == batch_dones.shape)
        batch_next_state_action_values = (1.0 - batch_dones) * batch_next_state_action_values
        # assert(batch_next_state_action_values.shape == batch_rewards.shape)
        batch_next_state_action_values = (GAMMA * batch_next_state_action_values + batch_rewards).detach()
        
        # assert(batch_state_action_values.shape == batch_next_state_action_values.shape)
        loss = ((batch_state_action_values - batch_next_state_action_values) * (batch_state_action_values - batch_next_state_action_values)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def train(self):
        for i in range(self.env_size):
            self.envs[i].reset()
        
        while self.steps_done < 10000000:
            batch_actions = self.get_batch_actions()
            for i in range(self.env_size):
                rnd = np.random.random()
                epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
                if rnd < epsilon:
                    action = np.random.choice(6)
                else:
                    action = batch_actions[i].item()
                state = get_state(self.envs[i].state())
                reward, done = self.envs[i].act(action)
                self.steps_done += 1
                self.scores[i] += reward
                next_state = get_state(self.envs[i].state())
                self.buffer.append((state, action, reward, next_state, done))

                if done:
                    self.envs[i].reset()
                    # print(self.scores[i])
                    self.scores[i] = 0
            
            if len(self.buffer) > self.batch_size:
                self.optimize()
                self.update_cnt += 1
                if self.update_cnt % 5 == 0:
                    self.target_q_network.load_state_dict(self.q_network.state_dict())
                    self.evaluate()
            
    def evaluate(self):
        test_env = Environment(env_name="breakout", random_seed=np.random.randint(0, 1<<30))
        score_sum = 0
        for i in range(10):
            test_env.reset()
            
            while True:
                state = torch.tensor(get_state(test_env.state()), device=device).float()
                action_values = self.q_network(state.unsqueeze(0))
                action = torch.max(action_values, dim=1).indices.item()
                reward, done = test_env.act(action)
                score_sum += reward
                if done:
                    break
        print(score_sum / 10)

if __name__ == "__main__":
    agent = DQNAgent()
    agent.train()