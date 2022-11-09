import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, in_channel_num, action_num):
        super(QNetwork, self).__init__()
        self.in_channel_num = in_channel_num
        self.action_num = action_num
        self.conv = nn.Conv2d(in_channel_num, 16, kernel_size=3, stride=1)
        def size_linear_unit(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        self.linear_unit_num = size_linear_unit(10) * size_linear_unit(10) * 16
        self.fc_hidden = nn.Linear(self.linear_unit_num, 64)
        self.fc_out = nn.Linear(64, action_num)
    
    def forward(self, x):
        h = F.relu(self.conv(x))
        h = h.view(h.size(0), -1)
        h = F.relu(self.fc_hidden(h))
        out = self.fc_out(h)
        return out