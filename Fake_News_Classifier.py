#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv(r'C:\Users\user\Downloads\fake-news\train.csv')


# In[3]:


df.head()


# In[4]:


x = df.drop('label', axis =1)


# In[5]:


x.head()


# In[6]:


y = df['label']


# In[7]:


df.shape


# In[8]:


df['text'][5]


# In[9]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer


# In[10]:


df = df.dropna()


# In[11]:


df.shape


# In[12]:


messages = df.copy()


# In[13]:


messages.head(10)


# In[14]:


messages.reset_index(inplace = True)


# In[15]:


messages.head(10)


# In[16]:


messages.drop(['index'], axis = 1, inplace = True)


# In[17]:


messages.head(10)


# In[19]:


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-z]', ' ', messages['title'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in  stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


# In[20]:


corpus[3]


# In[22]:


# Applying countvectorizer
# Creating Bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000, ngram_range=(1,3))
x = cv.fit_transform(corpus).toarray()


# In[23]:


x.shape


# In[24]:


y = messages['label']


# In[25]:


# Divide the datset into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y , test_size = 0.33, random_state = 2)


# In[27]:


cv.get_feature_names()[:20] # it's because of ngram_range


# In[28]:


cv.get_params()


# In[29]:


x


# In[31]:


count_df = pd.DataFrame(x_train, columns = cv.get_feature_names() )


# In[34]:


count_df.head()


# In[35]:


from matplotlib import pyplot as plt


# ### Training and Predicting the model

# In[36]:


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()


# In[37]:


from sklearn import metrics
import numpy as np
import itertools


# In[38]:


clf.fit(x_train, y_train)
preds = clf.predict(x_test)


# In[41]:


score  = metrics.accuracy_score(y_test, preds)
score


# In[42]:


import seaborn as sns


# In[44]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# In[46]:


cm = confusion_matrix(y_test, preds)
cm


# In[51]:


sns.set(font_scale = 1.5)
sns.heatmap(cm, annot = True, cmap = 'Blues')


# In[ ]:




