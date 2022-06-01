import numpy as np
from obs.metrics import loss_calc
import torch


class Environment:
    def __init__(self, art_embed, articles,  full_df, device,no_of_choices=10):
        '''
        art_embed: Products Embedding
        articles: Encoded Feature dataframe
        full_df: Main dataframe
        device: 'cpu'
        # no_of_choices: Total Number of output: 10 as we designed to pass 10 
        # input features and get 10 outputs individually
        '''
        
        self.art_embed = art_embed
        self.articles = articles
        self.list_article_index = articles.index.to_list()
        self.no_of_choices = no_of_choices
        self.train = full_df

        
    def reset(self):
        
        # sends random article_id's to display
        initial_choices = np.random.choice(self.list_article_index, self.no_of_choices)
        initial_choices_df = self.articles.loc[initial_choices]
        initial_choices_torch = torch.LongTensor(initial_choices_df.values).to('cpu', dtype=torch.float)
        
        for_page = self.train[self.train['article_id'].isin(initial_choices)]
        
        # send the dataframe with attributes to initial page and encoded data for model
        return for_page,initial_choices_torch      
    
    def step(self,list_of_clicked_arti):
        '''
        list of Selected output from previous episodes
        '''

        # Compute loss based on the number of items selected
        loss_compute = loss_calc(list_of_clicked_arti, device='cpu')
        reward = loss_compute.reward()
        
        # Choose Random articles for passing to model along with the selected one from previous episode
        if len(list_of_clicked_arti) <10:
            temp_num = 10-len(list_of_clicked_arti)
            for i in np.random.choice(self.articles.index.to_list(), temp_num):  # Pick out new products along with old ones to pass to model
                list_of_clicked_arti=np.append(list_of_clicked_arti, i)
        out_attri = self.articles.loc[list_of_clicked_arti] # outputs attributes of predicted ones, new ones and reward
        
        return torch.LongTensor(out_attri.values).to('cpu', dtype=torch.float), reward