import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.autograd import Variable
from itertools import count
import numpy as np

class PolicyNet_2(nn.Module):
    def __init__(self, kernel_size = 3):
        super(PolicyNet_2, self).__init__()
        assert(kernel_size%2)
        padding = (kernel_size - 1)//2
        self.conv1 = nn.Conv1d(1, 5, kernel_size, padding = padding)
        self.conv2 = nn.Conv1d(5, 5, kernel_size, padding = padding)
        self.conv3 = nn.Conv1d(5, 5, kernel_size, padding = padding)
        self.conv4 = nn.Conv1d(5, 2, kernel_size, padding = padding)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x


def get_points_modif(wp, initloc = np.array([0,0,0]), axis = 1):
    net = PolicyNet_2()
    net.load_state_dict(torch.load('TrainedModel'))
    wpz = np.reshape((wp-initloc)[:, axis], (1, 1, -1))
    res = net.forward(torch.from_numpy(wpz).float())
# return only mean
    res = res[:, 0, :]
    res = torch.clamp(res, -0.5, 0.5)
    wp = np.array(wp)
    wp[:, axis] += res.detach().numpy()[0]
    return wp

