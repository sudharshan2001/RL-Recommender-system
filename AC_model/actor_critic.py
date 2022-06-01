import torch

# Full Network
class ActorCriticNetwork(torch.nn.Module):
    def __init__(self, actor_model, critic_model):
        super(ActorCriticNetwork, self).__init__()
        '''
        actor_model: pass Actor Network
        critic_model: pass Critic Network
        '''
        
        self.actor = actor_model
        self.critic = critic_model
        
    def forward(self, state):
        
        actor = self.actor(state)
        critic = self.critic(state, actor)
        
        return actor, critic