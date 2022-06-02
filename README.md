# RL-Recommender-system
 Sequential recommendation based on Actor-Critic Algorithm

Dataset is taken from H&M Competition conducted in kaggle 
Download the dataset the place it in the same folder as main.py

Then install the requirements `pip install -r requirements.txt` 

To run the streamlit and manuallt select the products uncomment the specific lines mentioned in main.py and run `streamlit run main.py` to get to the main page

![front page](https://user-images.githubusercontent.com/72936645/171645618-3d9170a8-5541-4b16-8d1b-d6a51517b543.JPG)

(There might some error sometimes)

# Approach:
    This system is based on Actor-Critic algorithm.
    * First the products and selected features are embedded to matrix
    * The Actor Network outputs argmx of one of the products from the embeddings
    * The Critic Network will provide the reward for it and the reward is based on the products selected by the user 
    * Both the network will optimise themself in the long run
 

# Future update:
 Replace Streamlit with Ajax and Flask

