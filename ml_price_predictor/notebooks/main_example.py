#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import warnings
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# Use True if you dont have your own data
USE_DEMO_DATA = True

# Specify if you have your own data
TRAIN_DATA_PATH = 'YOUR_TRAIN_DATA_PATH'
TEST_DATA_PATH = 'YOUR_TEST_DATA_PATH'

if USE_DEMO_DATA:
    # making demo data for 2 categories
    df = pd.read_json('../data/dataset.json')
    cat_2_df = df.copy()
    cat_2_df['category_id'] = 2
    train_1, test_1 = train_test_split(df, shuffle=True, random_state=42, test_size=0.078)
    train_2, test_2 = train_test_split(cat_2_df, shuffle=True, random_state=42, test_size=0.078)
    
    train_df = pd.concat([train_1, train_2]).reset_index(drop=True)
    test_df = pd.concat([test_1, test_2]).reset_index(drop=True)
else:
    # reading your data
    train_df = pd.read_json(TRAIN_DATA_PATH)
    test_df = pd.read_json(TEST_DATA_PATH)
    
train_df.head()


# In[2]:


unique_categories = train_df['category_id'].unique()

print(unique_categories)


# #### Splitting data into separate catogories by category_id

# In[3]:


categories_dfs = {}

for category in tqdm(unique_categories):
    category_df = train_df[train_df['category_id'] == category].copy()
    categories_dfs[category] = category_df
    
categories_dfs.keys()


# ## Training models for each category

# In[4]:


from single_cat_model import SingleCategoryModel

categories_models = {}

for category in unique_categories:
    print("Training model for category", category, end='\n\n')
    model = SingleCategoryModel(category_number=category)
    
    model.train(categories_dfs[category])
    
    categories_models[category] = model


# ### Saving models to ONNX and CBM format

# In[5]:


categories_models


# In[6]:


for category in categories_models.keys():
    categories_models[category].export(output_path_onnx=f'./models/onnx/category_{category}_model.onnx')


# ## Validating model for each category

# ### Loading available models and validating on test data

# In[7]:


from single_cat_model import SingleCategoryModel
import pandas as pd
import numpy as np

# categries you saved your models for
categories = [1, 2]

test_categories_dfs = {}

# splitting test set to separate categories
for category in categories:
    test_categories_dfs[category] = test_df[test_df['category_id'] == category]
    
metrics_dict = {}

for category in categories:
    
    # initializing empty model
    model = SingleCategoryModel(category_number=category)
    # loading saved weights
    model.load_model(f'./models/onnx/category_{category}_model.onnx') # using path to CBM format!
    # validating on test set
    metrics = model.validate(
        valid_df=test_categories_dfs[category], 
        save_plot_path=f'./validation_plots/cat_{category}_pearson.png'
    )
    
    # saving metrics
    metrics_dict[category] = metrics
    
    print('\n')


# In[8]:


metrics_dict


# In[11]:


import matplotlib.pyplot as plt

plot = plt.imread('./validation_plots/cat_1_pearson.png')
plt.imshow(plot)


# ## Finetuning example

# In[12]:


model.finetune(test_1)

metrics = model.validate(test_1, './validation_plots/finetuned_model_plot.png')

print(metrics)


# In[13]:


plot = plt.imread('./validation_plots/finetuned_model_plot.png')
plt.imshow(plot)


# In[ ]:




