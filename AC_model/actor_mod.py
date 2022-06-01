import torch
from torch import nn
from torch.nn import functional as F

# Actor Model
class Actor(torch.nn.Module):
    def __init__(self, prod_embed, input_shape=7, n_rec=1, after_embed=30):
        '''
        prod_embed: Producte Embedding
        input_shape: Input Feature Length
        n_rec: Number of Recommendation
        after_embed: shape of column after embedding

        outputs : Prediction of an item from embedding
        '''
        
        super(Actor, self).__init__()
        
        self.input_shape = input_shape
        self.n_rec = n_rec
        self.after_embed = after_embed
        self.prod_embed = prod_embed
        
        no_fc1_dims = self.input_shape * 4
        no_fc2_dims = 2 * no_fc1_dims
        
        self.fc1_dims = nn.Linear(self.input_shape, no_fc1_dims)
        self.fc2_dims = nn.Linear(no_fc1_dims, no_fc2_dims)
        self.fc3_dims = nn.Linear(no_fc2_dims, self.n_rec * self.after_embed)
        
    def forward(self, state):
        '''
        state: input feature of songs
        '''

        x = F.relu(self.fc1_dims(state))
        x = F.relu(self.fc2_dims(x))
        x = F.relu(self.fc3_dims(x))
        
        # Reshape
        the_weights = x.view(-1, self.n_rec, self.after_embed)

        # Transpose
        product_trans = torch.transpose(self.prod_embed, 0, 1)

        # output the index with maximum weight
        argmax = torch.argmax(torch.matmul(the_weights, product_trans), dim=2)
        return argmax