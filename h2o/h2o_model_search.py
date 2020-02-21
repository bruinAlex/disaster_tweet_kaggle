#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
from scipy import sparse

import h2o
import matplotlib as plt

#Import the Estimators
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators import H2ORandomForestEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator

#Import h2o grid search 
import h2o.grid 
from h2o.grid.grid_search import H2OGridSearch


# ## Set h2o memory limit, cpus, and initialize instance

# In[5]:


h2o.init(nthreads = -1,max_mem_size = "12G")


# # Load the processed data

# In[6]:


X_train = sparse.load_npz("processed_data/raw_keyword_categorical_X_train_20k_feat.npz")
y_train = np.load("processed_data/raw_keyword_categorical_y_train.npy")
y_train = h2o.H2OFrame(y_train)
X_test = sparse.load_npz("processed_data/raw_keyword_categorical_test_processed_20k_feat.npz")


# ## Convert the numpy sparse matrices to an H2o dataframe

# In[7]:


X_train = h2o.H2OFrame(X_train)
X_test = h2o.H2OFrame(X_test)


# In[8]:


combined_train = X_train.cbind(y_train)


# # Start H2o experiment
# ## Create train, val, test splits

# In[9]:


train, valid, test = combined_train.split_frame([0.7, 0.15], seed=42)


# In[10]:


print("train:%d valid:%d test:%d" % (train.nrows, valid.nrows, test.nrows))


# In[11]:


y = "C110000"
ignore = ["C110000"]
x = list(set(train.names) - set(ignore))


# # Generalized Linear Model

# In[15]:


glm = H2OGeneralizedLinearEstimator(family = "binomial", seed=42, model_id = 'default_glm')


# # Train the model

# In[16]:


# get_ipython().run_cell_magic('time', '', 'glm.train(x = x, y = y, training_frame = train, validation_frame = valid)')
glm.train(x = x, y = y, training_frame = train, validation_frame = valid)

# In[17]:


glm


# In[18]:


glm.plot()


# In[19]:


# Save the model
default_glm_perf=glm.model_performance(valid)


# In[82]:


predictions = glm.predict(X_test)


# In[87]:


h2o.download_csv(predictions, "h2o_predictions.csv")


# # Random Forest

# In[33]:


rf = H2ORandomForestEstimator(seed=42, model_id='default_random_forest')
# get_ipython().run_line_magic('time', 'rf.train(x = x, y = y, training_frame = train, validation_frame = valid)')
rf.train(x = x, y = y, training_frame = train, validation_frame = valid)

# In[34]:


rf


# In[23]:


rf_predictions = rf.predict(X_test)
h2o.download_csv(rf_predictions, "h2o_rf_predictions.csv")


# # # Gradient Boosting Machine

# # In[24]:


# gbm= H2OGradientBoostingEstimator(seed=42, model_id='default_gbm')
# get_ipython().run_line_magic('time', 'gbm.train(x=x, y=y, training_frame=train, validation_frame = valid)')


# # In[25]:


# gbm


# # In[29]:


# gbm_predictions = gbm.predict(X_test)
# h2o.download_csv(gbm_predictions, "h2o_gbm_predictions.csv")


# # # Tune GBM with H2O GridSearch

# # In[35]:


# hyper_params = {'max_depth' : [1,3,5,6,7,8,9,10,12,13,15],
#                }

# gbm = H2OGradientBoostingEstimator(model_id='grid_gbm', ntrees=150,
#     seed=42
#     )

# gbm_grid = H2OGridSearch(gbm, hyper_params,
#                          grid_id = 'depth_gbm_grid',
#                          search_criteria = {
#                              "strategy":"Cartesian"})

# get_ipython().run_line_magic('time', 'gbm_grid.train(x=x, y=y, training_frame=train, validation_frame = valid)')


# # In[37]:


# sorted_gbm_depth = gbm_grid.get_grid(sort_by='rmsle',decreasing=True)
# sorted_gbm_depth


# # In[ ]:


# gbm = H2OGradientBoostingEstimator(max_depth=15, ntrees=80,
#     seed=42, model_id='tuned_gbm'
#     )
# get_ipython().run_line_magic('time', 'gbm.train(x=x, y=y, training_frame=train, validation_frame = valid)')
# gbm.plot(metric='auc')

