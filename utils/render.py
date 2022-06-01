import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import time

def render_page(df,width=100):
    list_df_images = df['path_to_image'].values
    list_df_name = df['prod_name'].values
    list_df_id =  df['article_id'].values

    
    list_of_opened_images = []
    for i in list_df_images:
        list_of_opened_images.append(Image.open(i))
    filteredImages = list_of_opened_images
    idx = 0 

    with st.form(str(time.time())):
        for _ in range(len(filteredImages)-1): 

            cols = st.columns([2,2,0.8,2,2]) 
            if idx < len(filteredImages): 
                cols[0].image(filteredImages[idx], width=width)
                cols[0].text(list_df_name[idx])
        
                rating1=cols[1].checkbox(f"Rating1",key=time.time())
                
            idx+=1
            
            if idx < len(filteredImages):
                cols[3].image(filteredImages[idx], width=width)
                cols[3].text(list_df_name[idx])
                rating2=cols[4].checkbox(f"Rating2",key=time.time())

            idx+=1

            cols = st.columns([2,2,0.8,2,2])  
            
            if idx < len(filteredImages): 
                cols[0].image(filteredImages[idx], width=width)
                cols[0].text(list_df_name[idx])
        
                rating3=cols[1].checkbox(f"Rating3",key=time.time())

            idx+=1
            
            if idx < len(filteredImages):
                cols[3].image(filteredImages[idx], width=width)
                cols[3].text(list_df_name[idx])

                rating4=cols[4].checkbox(f"Rating4",key=time.time())
                idx+=1
            else:
                break

            cols = st.columns([2,2,0.8,2,2]) 
            
            if idx < len(filteredImages): 
                cols[0].image(filteredImages[idx], width=width)
                cols[0].text(list_df_name[idx])

                rating5=cols[1].checkbox(f"Rating5",key=time.time())

            idx+=1
            
            if idx < len(filteredImages):
                cols[3].image(filteredImages[idx], width=width)
                cols[3].text(list_df_name[idx])

                rating6=cols[4].checkbox(f"Rating6",key=time.time())
            idx+=1
            
            cols = st.columns([2,2,0.8,2,2]) 
            
            if idx < len(filteredImages): 
                cols[0].image(filteredImages[idx], width=width)
                cols[0].text(list_df_name[idx])

                rating7=cols[1].checkbox(f"Rating7",key=time.time())

            idx+=1
            
            if idx < len(filteredImages):
                cols[3].image(filteredImages[idx], width=width)
                cols[3].text(list_df_name[idx])

                rating8=cols[4].checkbox(f"Rating8",key=time.time())
            idx+=1

            cols = st.columns([2,2,0.8,2,2]) 
            
            if idx < len(filteredImages): 
                cols[0].image(filteredImages[idx], width=width)
                cols[0].text(list_df_name[idx])

                rating9=cols[1].checkbox(f"Rating9",key=time.time())

            idx+=1
            
            if idx < len(filteredImages):
                cols[3].image(filteredImages[idx], width=width)
                cols[3].text(list_df_name[idx])

                rating10=cols[4].checkbox(f"Rating10",key=time.time())
                idx+=1
            else:
                break
            
        # submit = st.button('Submit',key=time.time())
        # # donee = st.button('End',key=time.time())
        ratings=[rating1, rating2,rating3, rating4,rating5, rating6,rating7, rating8,rating9, rating10]

        # if submit:
        #     return list_df_id[ratings]

        submitted = st.form_submit_button("Submit")

        if submitted:
            # print(list_df_id[ratings])
            return list_df_id[ratings]
            
            
            

            
            