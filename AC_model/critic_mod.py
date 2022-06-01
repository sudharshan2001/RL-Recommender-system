import torch
from torch import nn
from torch.nn import functional as F

# Critic Network
class Critic(torch.nn.Module):
    def __init__(self, input_shape=7, actor_output=1):
        super(Critic, self).__init__()
        '''
        input_shape: Number of Features 
        actor_output: Number of Output from Actor Network
        '''
        
        self.input_shape = input_shape
        self.actor_output = actor_output
        
        no_fc1_dims = 4*(self.input_shape + self.actor_output)
        no_fc2_dims = no_fc1_dims
        output_fc = 1
        
        self.fc1_dims = nn.Linear(self.input_shape + self.actor_output, no_fc1_dims)
        self.fc2_dims = nn.Linear(no_fc1_dims, no_fc2_dims)
        self.fc3_dims = nn.Linear(no_fc2_dims, output_fc)
        
    def forward(self, x, actor_output_prev):
        '''
        X: Features
        actor_output_prev: Output from Actor Network
        '''

        x = torch.cat((x, actor_output_prev), -1)
        x = F.relu(self.fc1_dims(x))
        x = F.relu(self.fc2_dims(x))
        x = self.fc3_dims(x)
        return x