import torch
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer, transition
from model import QNetwork
from minatar import Environment

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_state(obs):
    return (torch.tensor(obs, device=device).permute(2,0,1)).unsqueeze(0).float()

class DQNAgent(object):
    def __init__(self, env):
        self.policy_net = QNetwork(env.state_shape()[2], env.num_actions())
        self.target_net = QNetwork(env.state_shape()[2], env.num_actions())
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.to(device)
        self.target_net.to(device)
        self.env = env
        self.buffer = ReplayBuffer(10000)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.num_actions = 6
        self.t = 0
        self.episode_counter = 0
        self.update_counter = 0
        self.NUM_FRAMES = 1000000
        self.REPLAY_START_SIZE = 5000
        self.BATCH_SIZE = 64
        self.TARGET_NETWORK_UPDATE_FREQ = 1000

    def select_action(self, state):
        if self.t < self.REPLAY_START_SIZE:
            action = torch.tensor([[np.random.randint(0, self.num_actions)]], device=device)
        else:
            if np.random.random() < 0.1:
                action = torch.tensor([[np.random.randint(0, self.num_actions)]], device=device)
            else:
                with torch.no_grad():
                    action = self.policy_net(state).max(1)[1].view(1,1)
        return action

    def dynamics(self, action):
        reward, terminated = self.env.act(action.item())
        next_state = get_state(self.env.state())
        return next_state, torch.tensor([[reward]], device=device).float(), torch.tensor([[terminated]], device=device)
        
    def update(self):
        if self.t <= self.REPLAY_START_SIZE or len(self.buffer) < self.BATCH_SIZE:
            return

        sample = self.buffer.sample(self.BATCH_SIZE)
        batch_samples = transition(*zip(*sample))
        states = torch.cat(batch_samples.state)
        next_states = torch.cat(batch_samples.next_state)
        actions = torch.cat(batch_samples.action)
        rewards = torch.cat(batch_samples.reward)
        is_terminal = torch.cat(batch_samples.is_terminal)

        Q_s_a = self.policy_net(states).gather(1, actions)
        none_terminal_next_state_index = torch.tensor([i for i, is_term in enumerate(is_terminal) if is_term == 0], dtype=torch.int64, device=device)
        none_terminal_next_states = next_states.index_select(0, none_terminal_next_state_index)
        Q_next_s_next_a = torch.zeros(len(sample), 1, device=device)
        if len(none_terminal_next_states) != 0:
            Q_next_s_next_a[none_terminal_next_state_index] = self.target_net(none_terminal_next_states).detach().max(1)[0].unsqueeze(1)
        
        target = rewards + Q_next_s_next_a
        loss = F.smooth_l1_loss(target, Q_s_a)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_counter += 1

    def train(self):

        while self.t < self.NUM_FRAMES:
            self.env.reset()
            G = 0.0
            is_terminated = False
            state = get_state(self.env.state())
            while (not is_terminated) and self.t < self.NUM_FRAMES:
                action = self.select_action(state)
                next_state, reward, is_terminated = self.dynamics(action)
                self.buffer.add(state, next_state, action, reward, is_terminated)
                self.update()

                if self.update_counter > 0 and self.update_counter % self.TARGET_NETWORK_UPDATE_FREQ == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

                G += reward.item()
                self.t += 1
                state = next_state
            print(G)
            self.episode_counter += 1
    
if __name__ == '__main__':
    env = Environment('breakout')
    dqn = DQNAgent(env=env)
    dqn.train()


