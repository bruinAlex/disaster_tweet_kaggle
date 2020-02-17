#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd

import regex as re

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split#, cross_val_score
from sklearn import metrics, preprocessing

import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from spacy.matcher import Matcher

from joblib import dump, load


# In[2]:


spacy.prefer_gpu()


# In[3]:


# !python -m spacy download en_core_web_sm

# nlp = spacy.load("en_core_web_sm")
nlp = English()


# In[4]:


# Load English tokenizer, tagger, parser, NER and word vectors
parser = English()


# In[5]:


# Create a blank Tokenizer with just the English vocab
tokenizer = Tokenizer(nlp.vocab)


# In[6]:


stop_words = spacy.lang.en.stop_words.STOP_WORDS


# In[7]:


# Create matcher for hashtags
matcher = Matcher(nlp.vocab)
matcher.add('HASHTAG', None, [{'ORTH': '#'}, {'IS_ASCII': True}])


# In[8]:


# Custom Transformer that extracts columns passed as argument to its constructor 
class FeatureSelector( BaseEstimator, TransformerMixin ):
    # Class Constructor 
    def __init__(self, feature_names):
        self.feature_names = feature_names 
    
    # Return self nothing else to do here    
    def fit(self, X, y = None):
        return self 
    
    # Method that describes what we need this transformer to do
    # This one pulls up the list of feature columns you pass in and returns just those columns
    def transform(self, X, y = None):
        return X[self.feature_names] 


# In[9]:


# Custom transformer that takes in a string and returns new categorical features
class CategoricalTextTransformer(BaseEstimator, TransformerMixin):
    # Class constructor method that takes in a list of values as its argument
    def __init__(self):
        self.hashtag_pattern = re.compile("(?:^|\s)[＃#]{1}(\w+)", re.UNICODE)
        
        
    # Return self nothing else to do here
    def fit(self, X, y = None):
        return self
    
    
    # Test helper func to just return the text in all lower case
    def is_lower(self, obj):
        if obj.islower():
            return 1
        else:
            return 0
    
    
    def is_upper(self, obj):
        if obj.isupper():
            return 1
        else:
            return 0

                
    # Transformer method to take in strings from a dataframe and return some extra features
    def transform(self, X , y = None):
        # Copy the incoming df to prevent setting on copy errors
        X = X.copy()
        
        # Return binary indicator of whether tweet is all lowercase
        X['is_lower'] = X['text'].apply(self.is_lower)
        
        # Return binary indicator of whether tweet is all uppercase
        X['is_upper'] = X['text'].apply(self.is_upper)
    
        # Drop original text col
        # The only thing remaining now will be the lowercased text
        X = X.drop('text', axis=1)
        
        # returns numpy array
        return X.values 
    
    
    # Transformer method to take in strings from a dataframe and return some extra features
    def fit_transform(self, X , y = None):
        # Copy the incoming df to prevent setting on copy errors
        X = X.copy()
        
        # Return binary indicator of whether tweet is all lowercase
        X['is_lower'] = X['text'].apply(self.is_lower)
        
        # Return binary indicator of whether tweet is all uppercase
        X['is_upper'] = X['text'].apply(self.is_upper)
        
        # Drop original text col
        # The only thing remaining now will be the lowercased text
        X = X.drop('text', axis=1)
        
        # returns numpy array
        return X.values 
    
    


# In[10]:


# Custom transformer processes the keyword feature as a categorical
class CategoricalTransformer(BaseEstimator, TransformerMixin):
    # Class constructor method that takes in a list of values as its argument
    def __init__(self):
        self.ohe_model = preprocessing.OneHotEncoder(handle_unknown='error',
                                         drop='first',
                                         sparse=False)

        
    # Return self nothing else to do here
    def fit(self, X, y = None):
        return self
    

    # Transformer method to take in strings from a dataframe and return some extra features
    def transform(self, X , y = None):
        # Copy the incoming df to prevent setting on copy errors
        X = X.copy()
    
        # Fill NaNs with "None"
        # Missing values will cause the one-hot encoding to fail
        X = X.fillna("none")
        
        X = self.ohe_model.transform(X)
        
        return X
        
        
    # Transformer method to take in strings from a dataframe and return some extra features
    def fit_transform(self, X , y = None):
        # Copy the incoming df to prevent setting on copy errors
        X = X.copy()
        
        # Fill NaNs with "None"
        # Missing values will cause the one-hot encoding to fail
        X = X.fillna("none")
        
        X = self.ohe_model.fit_transform(X)
        

        return X 


# In[11]:


class DenseTfidfVectorizer(TfidfVectorizer):
    def __init__(self):
        self.tfidf_model = TfidfVectorizer(tokenizer=self.spacy_tokenizer)
        
    def spacy_tokenizer(self, obj):
        doc = nlp(obj)

        # Looks for hashtags
        matches = matcher(doc)
        spans = []
        for match_id, start, end in matches:
            spans.append(doc[start:end])

        for span in spans:
            span.merge()

        return [t.text.lower() for t in doc if t not in stop_words and not t.is_punct | t.is_space]
        
    def transform(self, raw_documents):
        X = self.tfidf_model.transform(raw_documents['text'])

        return X.toarray() # Changes the scipy sparse array to a numpy matrix

    
    def fit_transform(self, raw_documents, y=None):
        X = self.tfidf_model.fit_transform(raw_documents['text'], y=y)

        return X.toarray()


# In[12]:


# Categorical text features
cat_text_features = ['text']

# Text features for text pipeline
text_features = ['text']

# Define categorical pipeline
cat_text_pipeline = Pipeline(
    steps = [('cat_text_selector', FeatureSelector(cat_text_features)),
             ('cat_text_transformer', CategoricalTextTransformer()),
            ],
    verbose = True
)

# Define the text training pipeline
text_pipeline = Pipeline(
    steps = [('text_selector', FeatureSelector(text_features)),
             ('text_tfidf', DenseTfidfVectorizer())
            ],
    verbose = True
)


# # Model 1
# - tf-idf
# - text is upper
# - text is lower

# In[13]:


# Combine all our pipelines into a single one inside the FeatureUnion object
# Right now we only have one pipeline which is our text one
full_pipeline = FeatureUnion(
    transformer_list=[
        ('text_pipeline', text_pipeline),
        ('cat_text_pipeline', cat_text_pipeline),
                     ]
)


# In[14]:


train_df = pd.read_csv("data/train.csv")


# In[15]:


X_train = train_df.copy()
y_train = X_train.pop('target').values


# In[17]:


# Process text and categorical features
X_train_processed = full_pipeline.fit_transform(X_train)


# In[51]:


# %%time
# lrcv =  LogisticRegressionCV(cv=10, 
#                              max_iter = 4000,
#                              random_state=42, 
#                              n_jobs=-1,
#                              scoring = 'f1',
#                             )
# lrcv.fit(X_train_processed, y_train)


# In[52]:


# lrcv.scores_[1].mean(axis=0).max()


# In[59]:


# # Save the model to disk
# dump(lrcv, 'saved_models/model_01.joblib') 

# # # To load it later:
# # model_01 = load('saved_models/model_01.joblib') 


# # Get test predictions for kaggle scoring

# In[62]:


# %%time
# test_df = pd.read_csv('data/test.csv')

# # Preprocess test data
# test_processed = full_pipeline.transform(test_df)

# test_predictions = lrcv.predict(test_processed)
# test_id = test_df['id']
# test_predictions_df = pd.DataFrame([test_id, test_predictions]).T
# test_predictions_df.columns = ['id', 'target']
# test_predictions_df.to_csv('test_preds.csv', index=False)


# ## Create dict for model results

# In[63]:


# model_results = dict()
# model_results['model_01'] = {'best mean kfold score' : lrcv.scores_[1].mean(axis=0).max(), 
#                              'kaggle submission score' : 0.80777
#                             }


# In[64]:


# model_results


# # Model 2
# - tf-idf
# - text is upper
# - text is lower
# - include hashtag counts
# - include keywords as categorical

# In[68]:


# # Custom transformer that takes in a string and returns new categorical features
# class CategoricalTextTransformer(BaseEstimator, TransformerMixin):
#     # Class constructor method that takes in a list of values as its argument
#     def __init__(self):
#         self.hashtag_pattern = re.compile("(?:^|\s)[＃#]{1}(\w+)", re.UNICODE)
        
        
#     # Return self nothing else to do here
#     def fit(self, X, y = None):
#         return self
    
    
#     # Test helper func to just return the text in all lower case
#     def is_lower(self, obj):
#         if obj.islower():
#             return 1
#         else:
#             return 0
    
    
#     def is_upper(self, obj):
#         if obj.isupper():
#             return 1
#         else:
#             return 0


#     def count_hashtags(self, obj):
#         hashtag_count = len(re.findall(self.hashtag_pattern, obj))
#         return hashtag_count
        
        
#     # Transformer method to take in strings from a dataframe and return some extra features
#     def transform(self, X , y = None):
#         # Copy the incoming df to prevent setting on copy errors
#         X = X.copy()
        
#         # Return binary indicator of whether tweet is all lowercase
#         X['is_lower'] = X['text'].apply(self.is_lower)
        
#         # Return binary indicator of whether tweet is all uppercase
#         X['is_upper'] = X['text'].apply(self.is_upper)
    
#         # Count the number of hashtags in the text
#         X['hashtag_count'] = X['text'].apply(self.count_hashtags)
    
#         # Drop original text col
#         # The only thing remaining now will be the lowercased text
#         X = X.drop('text', axis=1)
        
#         # returns numpy array
#         return X.values 
    
    
#     # Transformer method to take in strings from a dataframe and return some extra features
#     def fit_transform(self, X , y = None):
#         # Copy the incoming df to prevent setting on copy errors
#         X = X.copy()
        
#         # Return binary indicator of whether tweet is all lowercase
#         X['is_lower'] = X['text'].apply(self.is_lower)
        
#         # Return binary indicator of whether tweet is all uppercase
#         X['is_upper'] = X['text'].apply(self.is_upper)
        
#         # Count the number of hashtags in the text
#         X['hashtag_count'] = X['text'].apply(self.count_hashtags)
        
#         # Drop original text col
#         # The only thing remaining now will be the lowercased text
#         X = X.drop('text', axis=1)
        
#         # returns numpy array
#         return X.values 
    
    


# In[69]:


# # Categorical text features
# cat_text_features = ['text']

# # Text features for text pipeline
# text_features = ['text']

# # Categorical features for text pipeline
# cat_features = ['keyword']

# # Define categorical pipeline
# cat_text_pipeline = Pipeline(
#     steps = [('cat_text_selector', FeatureSelector(cat_text_features)),
#              ('cat_text_transformer', CategoricalTextTransformer()),
#             ],
#     verbose = True
# )

# # Define the text training pipeline
# text_pipeline = Pipeline(
#     steps = [('text_selector', FeatureSelector(text_features)),
# #              ('text_transformer', TextTokenizerTransformer()),
#              ('text_tfidf', DenseTfidfVectorizer())
#             ],
#     verbose = True
# )

# # Define the keyword categorical training pipeline
# cat_pipeline = Pipeline(
#     steps = [('cat_selector', FeatureSelector(cat_features)),
#              ('cat_transformer', CategoricalTransformer())
#             ],
#     verbose = True
# )


# In[70]:


# # Combine all our pipelines into a single one inside the FeatureUnion object
# # Right now we only have one pipeline which is our text one
# full_pipeline = FeatureUnion(
#     transformer_list=[
#         ('cat_pipeline', cat_pipeline),
#         ('text_pipeline', text_pipeline),
#         ('cat_text_pipeline', cat_text_pipeline),
#                      ]
# )


# In[71]:


# %%time
# # Process text and categorical features
# X_train_processed = full_pipeline.fit_transform(X_train)

# lrcv02 =  LogisticRegressionCV(cv=10, 
#                              max_iter = 4000,
#                              random_state=42, 
#                              n_jobs=-1,
#                              scoring = 'f1',
#                             )

# lrcv02.fit(X_train_processed, y_train)


# In[72]:


# lrcv02.scores_[1].mean(axis=0).max()

# # Save the model to disk
# dump(lrcv02, 'saved_models/model_02.joblib') 

# # # To load it later:
# # model_02 = load('saved_models/model_02.joblib') 


# In[73]:


# %%time
# # Preprocess test data
# test_processed = full_pipeline.transform(test_df)

# test_predictions = lrcv02.predict(test_processed)
# test_id = test_df['id']
# test_predictions_df = pd.DataFrame([test_id, test_predictions]).T
# test_predictions_df.columns = ['id', 'target']
# test_predictions_df.to_csv('test_preds_02.csv', index=False)


# In[74]:


# model_results['model_02'] = {'best mean kfold score' : lrcv02.scores_[1].mean(axis=0).max(), 
#                              'kaggle submission score' : 0.79243
#                             }
# model_results


# # Model 3
# - Only TF-IDF on tweet

# In[75]:


# # Text features for text pipeline
# text_features = ['text']

# # Define the text training pipeline
# text_pipeline = Pipeline(
#     steps = [('text_selector', FeatureSelector(text_features)),
#              ('text_tfidf', DenseTfidfVectorizer())
#             ],
#     verbose = True
# )

# # Combine all our pipelines into a single one inside the FeatureUnion object
# # Right now we only have one pipeline which is our text one
# full_pipeline = FeatureUnion(
#     transformer_list=[
#         ('text_pipeline', text_pipeline),
#                      ]
# )


# In[76]:


# %%time
# # Process text and categorical features
# X_train_processed = full_pipeline.fit_transform(X_train)

# lrcv03 =  LogisticRegressionCV(cv=10, 
#                              max_iter = 4000,
#                              random_state=42, 
#                              n_jobs=-1,
#                              scoring = 'f1',
#                             )

# lrcv03.fit(X_train_processed, y_train)

# lrcv03.scores_[1].mean(axis=0).max()

# # Save the model to disk
# dump(lrcv03, 'saved_models/model_03.joblib') 

# # # To load it later:
# # model_03 = load('saved_models/model_03.joblib') 


# In[78]:


# %%time
# # Preprocess test data
# test_processed = full_pipeline.transform(test_df)

# test_predictions = lrcv03.predict(test_processed)
# test_id = test_df['id']
# test_predictions_df = pd.DataFrame([test_id, test_predictions]).T
# test_predictions_df.columns = ['id', 'target']
# test_predictions_df.to_csv('test_preds_03.csv', index=False)


# In[80]:


# model_results['model_03'] = {'best mean kfold score' : lrcv03.scores_[1].mean(axis=0).max(), 
#                              'kaggle submission score' : 0.79959
#                             }
# model_results


# # Model 4
# - TF-IDF
# - text is upper
# - text is lower
# - keywords as categorical

# In[84]:


# # Custom transformer that takes in a string and returns new categorical features
# class CategoricalTextTransformer(BaseEstimator, TransformerMixin):
#     # Class constructor method that takes in a list of values as its argument
#     def __init__(self):
#         self.hashtag_pattern = re.compile("(?:^|\s)[＃#]{1}(\w+)", re.UNICODE)
        
        
#     # Return self nothing else to do here
#     def fit(self, X, y = None):
#         return self
    
    
#     # Test helper func to just return the text in all lower case
#     def is_lower(self, obj):
#         if obj.islower():
#             return 1
#         else:
#             return 0
    
    
#     def is_upper(self, obj):
#         if obj.isupper():
#             return 1
#         else:
#             return 0
        
        
#     # Transformer method to take in strings from a dataframe and return some extra features
#     def transform(self, X , y = None):
#         # Copy the incoming df to prevent setting on copy errors
#         X = X.copy()
        
#         # Return binary indicator of whether tweet is all lowercase
#         X['is_lower'] = X['text'].apply(self.is_lower)
        
#         # Return binary indicator of whether tweet is all uppercase
#         X['is_upper'] = X['text'].apply(self.is_upper)
    
#         # Drop original text col
#         # The only thing remaining now will be the lowercased text
#         X = X.drop('text', axis=1)
        
#         # returns numpy array
#         return X.values 
    
    
#     # Transformer method to take in strings from a dataframe and return some extra features
#     def fit_transform(self, X , y = None):
#         # Copy the incoming df to prevent setting on copy errors
#         X = X.copy()
        
#         # Return binary indicator of whether tweet is all lowercase
#         X['is_lower'] = X['text'].apply(self.is_lower)
        
#         # Return binary indicator of whether tweet is all uppercase
#         X['is_upper'] = X['text'].apply(self.is_upper)
        
#         # Drop original text col
#         # The only thing remaining now will be the lowercased text
#         X = X.drop('text', axis=1)
        
#         # returns numpy array
#         return X.values 
    
    


# In[85]:


# # Categorical text features
# cat_text_features = ['text']

# # Text features for text pipeline
# text_features = ['text']

# # Categorical features for text pipeline
# cat_features = ['keyword']

# # Define categorical pipeline
# cat_text_pipeline = Pipeline(
#     steps = [('cat_text_selector', FeatureSelector(cat_text_features)),
#              ('cat_text_transformer', CategoricalTextTransformer()),
#             ],
#     verbose = True
# )

# # Define the text training pipeline
# text_pipeline = Pipeline(
#     steps = [('text_selector', FeatureSelector(text_features)),
#              ('text_tfidf', DenseTfidfVectorizer())
#             ],
#     verbose = True
# )

# # Define the keyword categorical training pipeline
# cat_pipeline = Pipeline(
#     steps = [('cat_selector', FeatureSelector(cat_features)),
#              ('cat_transformer', CategoricalTransformer())
#             ],
#     verbose = True
# )


# In[86]:


# # Combine all our pipelines into a single one inside the FeatureUnion object
# # Right now we only have one pipeline which is our text one
# full_pipeline = FeatureUnion(
#     transformer_list=[
#         ('cat_pipeline', cat_pipeline),
#         ('text_pipeline', text_pipeline),
#         ('cat_text_pipeline', cat_text_pipeline),
#                      ]
# )


# In[87]:


# %%time
# # Process text and categorical features
# X_train_processed = full_pipeline.fit_transform(X_train)

# lrcv04 =  LogisticRegressionCV(cv=10, 
#                              max_iter = 4000,
#                              random_state=42, 
#                              n_jobs=-1,
#                              scoring = 'f1',
#                             )

# lrcv04.fit(X_train_processed, y_train)

# print(lrcv04.scores_[1].mean(axis=0).max())

# # Save the model to disk
# dump(lrcv04, 'saved_models/model_04.joblib') 

# # # To load it later:
# # model_03 = load('saved_models/model_03.joblib') 


# In[88]:


# %%time
# # Preprocess test data
# test_processed = full_pipeline.transform(test_df)

# test_predictions = lrcv04.predict(test_processed)
# test_id = test_df['id']
# test_predictions_df = pd.DataFrame([test_id, test_predictions]).T
# test_predictions_df.columns = ['id', 'target']
# test_predictions_df.to_csv('test_preds_04.csv', index=False)


# In[89]:


# model_results['model_04'] = {'best mean kfold score' : lrcv04.scores_[1].mean(axis=0).max(), 
#                              'kaggle submission score' : 0.79345
#                             }
# model_results


# # Model 5: LinearSVC

# In[40]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform


# In[39]:


# from sklearn.svm import LinearSVC


# In[42]:


# svc_clf = LinearSVC(verbose = 1,
#                     random_state = 42,
#                    )

# distributions = dict(C=uniform(loc=0, scale=4))


# In[43]:


# clf = RandomizedSearchCV(svc_clf, 
#                          distributions, 
#                          random_state=42,
#                          verbose=1,
#                          n_jobs=-1,
#                          scoring='f1',
#                          cv=10,
# #                          n_iter=60,
#                         )


# In[44]:


# %%time
# search = clf.fit(X_train_processed, y_train)


# In[45]:


# search.best_params_


# In[46]:


# search.cv_results_


# In[ ]:


# # Preprocess test data
# test_processed = full_pipeline.transform(test_df)


# In[ ]:





# # Model 6: Random Forest
# - Use pipeline from Model 1 which had highest score for logistic regression

# In[18]:


from sklearn.model_selection import RandomizedSearchCV


# In[20]:


from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=100, 
                                random_state=42, 
                                verbose=1, 
                                n_jobs=2
                               ) # Using all CPUs leads to memory crashes


# In[21]:


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]# Create the random grid

distributions = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# In[22]:


clf = RandomizedSearchCV(rf_clf, 
                         distributions, 
                         random_state=42,
                         verbose=1,
                         n_jobs=1,
                         scoring='f1',
                         cv=7,
                         n_iter=60,
                        )


# In[23]:


search = clf.fit(X_train_processed, y_train)


# In[ ]:





# In[33]:


# Save the model to disk
dump(search.best_estimator_, 'saved_models/model_06.joblib') 


# In[34]:


test_df = pd.read_csv('data/test.csv')

# Preprocess test data
test_processed = full_pipeline.transform(test_df)


# In[35]:


# Preprocess test data
test_processed = full_pipeline.transform(test_df)

test_predictions = search.predict(test_processed)
test_id = test_df['id']
test_predictions_df = pd.DataFrame([test_id, test_predictions]).T
test_predictions_df.columns = ['id', 'target']
test_predictions_df.to_csv('test_preds_06.csv', index=False)


# In[ ]:




