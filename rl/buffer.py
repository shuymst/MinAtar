import random
from collections import namedtuple

transition = namedtuple('transition', 'state, next_state, action, reward, is_terminal')

class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.location = 0
        self.buffer = []
    
    def __len__(self):
        return len(self.buffer)
    
    def add(self, *args):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition(*args))
        else:
            self.buffer[self.location] = transition(*args)
        self.location = (self.location + 1) % self.buffer_size
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)