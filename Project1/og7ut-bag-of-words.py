
# coding: utf-8

# read data

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


trn = open("trn_data.txt").read().strip().split("\n")
trn_label = open("trn_label.txt").read().strip().split("\n")
trn_label_int = []
for label in trn_label:
    trn_label_int.append(int(label))


# In[3]:


dev = open("dev_data.txt").read().strip().split("\n")
dev_label = open("dev_label.txt").read().strip().split("\n") 
dev_label_int = []
for label in dev_label:
    dev_label_int.append(int(label))


# In[4]:


tst = open("tst_data.txt").read().strip().split("\n")


# In[5]:


from nltk.corpus import stopwords, words

stop_words = set(stopwords.words('english'))
words_list = set(words.words())
main_list = []
for review in trn:
    review_split = review.split()
    for word in review_split:
        if word.isalpha():
            if word.lower() not in stop_words:
                if word.lower() in words_list:
                    main_list.append(word.lower())


# In[6]:


dataframe_words = pd.DataFrame(main_list)


# In[7]:


dataframe_words.columns= ["word"]


# In[8]:


count = dataframe_words.groupby('word')


# In[9]:


df=pd.DataFrame(count.size())
df.columns= ["frequency"]


# In[10]:


for index, row in df.iterrows():
    if row['frequency']<4 or row['frequency']>5000:
        df.drop(index, inplace=True)


# In[11]:


vector=[]
for index, row in df.iterrows():
       vector.append(index)
vector


# In[12]:


train_data=[]
a=1
b=0
for review in trn:
    review_split = review.split()
    rows=[]
    for word in vector:
        if word in review_split:
            rows.append(a)
        else:
            rows.append(b)
    train_data.append(rows)     


# In[13]:


dev_data=[]
a=1
b=0
for review in dev:
    review_split = review.split()
    rows=[]
    for word in vector:
        if word in review_split:
            rows.append(a)
        else:
            rows.append(b)
    dev_data.append(rows) 

