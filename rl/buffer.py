import random

class ReplayBuffer(object):
    def __init__(self, capacity):
        self._storage = []
        self._capacity = capacity
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)
    
    def add(self, state, action, reward, next_states, done):
        data = (state, action, reward, next_states, done)
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._capacity
    
    def _encode_samples(self, indexs):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in indexs:
            data = self._storage[i]
            state, action, reward, next_state, done = data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        return states, actions, rewards, next_states, dones

    def sample(self, batch_size):
        indexs = random.