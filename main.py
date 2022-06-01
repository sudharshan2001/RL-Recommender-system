import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from utils.render import render_page
from sklearn import preprocessing
import torch, copy, random
from AC_model import actor_critic, actor_mod, critic_mod
from AC_model.actor_critic import ActorCriticNetwork
from AC_model.actor_mod import Actor
from AC_model.critic_mod import Critic
from obs.environment import Environment
from obs.metrics import loss_calc
from utils.parametrs import CFG

train = pd.read_csv(CFG.path_to_csv)

train_df = train.copy()

# Generate the path for images
def dummy(x):
    return './images/0' + str(x)[:2] +'/0'+ str(x)+'.jpg'

train['path_to_image'] = train['article_id'].apply(dummy)

# Drop unwanted columns
train_df.drop(['prod_name', 'product_type_name', 'graphical_appearance_name', 'product_group_name'
              , 'colour_group_name', 'perceived_colour_value_name','perceived_colour_value_id',
              'perceived_colour_master_id','perceived_colour_master_name', 'department_no', 'department_name'
              , 'index_name', 'index_group_name', 'section_name', 'garment_group_name', 'detail_desc','product_code'], axis=1, inplace=True)
train_df.set_index('article_id', inplace=True)

# Encode Label to pass it to NN
le = preprocessing.LabelEncoder()

label_object = {}
categorical_columns = train_df.columns.to_list()
for col in categorical_columns:
    labelencoder =  preprocessing.LabelEncoder()
    labelencoder.fit(train_df[col])
    train_df[col] = labelencoder.fit_transform(train_df[col])
    label_object[col] = labelencoder

device=CFG.device
after_embedding = CFG.no_embedding

# Embedding Products
art_embed = torch.randn(len(train_df.index), after_embedding)
art_embed = art_embed.to(device, dtype=torch.float)

actor = Actor(prod_embed=art_embed)
critic = Critic()
model = ActorCriticNetwork(actor, critic)

actor.to(device)
critic.to(device)
model.to(device)

model2 = copy.deepcopy(model)
model2.to(device)
model2.load_state_dict(model.state_dict())

# Initialising Environment
env = Environment(art_embed=art_embed, articles=train_df, full_df=train, device=device)

# Parsing original DF from output of actor network
def pred_to_df(list_pred):
    temp_detached=[]
    for i in list_pred:
        temp_detached.append(i[0])
    predicted_article_id_img = train.iloc[temp_detached]

    return predicted_article_id_img


loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=CFG.learning_rate)

losses = []
gamma = CFG.gamma
i=0

# initial state (Random Features)
for_page, state1 = env.reset()

while True:
        
        act_, crit_ = model(state1.detach())

        for_page = pred_to_df(act_)

        '''
        The next two snippet will choose random product from the output products
        you can comment them out and uncomment the render_page to access and select via
        streamlit but its kinda messy
        '''

        random_select =np.random.choice([True, False],CFG.total_no_of_pred)
        selected_items = for_page.article_id[random_select].values #select random choices (by user)
        # selected_items = render_page(for_page)
        state2, reward = env.step(selected_items) 
        state1 = state2

        with torch.no_grad():
            _, crti_2 = model2(state2.detach())

        REWARD = reward + gamma * crti_2

        loss = loss_fn(crit_, REWARD.detach())
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        losses.append(loss.item())
        optimizer.step()

        # Break after 1000 iterations
        if i > 1000:
            break


losses = np.array(losses)

