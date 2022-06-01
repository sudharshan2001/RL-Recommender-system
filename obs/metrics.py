import torch
from torch import nn
from torch.nn import functional as F

class loss_calc:
    
    def __init__(self,list_of_clicked_arti,device, alpha=0.5, reward_per_correct_pred=10,no_of_recommendation=10):
        '''
        list_of_clicked_arti:
        reward_per_correct_pred: Reward for Correct Prediction, Penalty will be negative of it
        no_of_recommendation: Total number of prediction (per episode)
        '''
        
        self.list_of_clicked_arti = list_of_clicked_arti
        self.no_of_clicked_arti = len(list_of_clicked_arti)
        self.reward_per_correct_pred = reward_per_correct_pred
        self.no_of_recommendation = no_of_recommendation
        
    def reward(self):
        
        reward_for_correct = self.no_of_clicked_arti * self.reward_per_correct_pred 
        penal_for_wrong = (self.no_of_recommendation - self.no_of_clicked_arti) * -self.reward_per_correct_pred 
        
        final_reward = reward_for_correct + penal_for_wrong
        reward = torch.Tensor([final_reward]).to('cpu', dtype=torch.float)
        
#         penal = torch.Tensor(penal_for_wrong).to(device, dtype=torch.float)
#         reward.item()
        return reward
