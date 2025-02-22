#%%
import pandas as pd
#%%
df = pd.read_parquet('data/ember.parquet', engine='fastparquet')

#%% 

small = df.head()