
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np


# In[7]:


trn = open("trn_data.txt").read().strip().split("\n")
trn_label = open("trn_label.txt").read().strip().split("\n")
trn_label_int = []
for label in trn_label:
    trn_label_int.append(int(label))


# In[8]:


dev = open("dev_data.txt").read().strip().split("\n")
dev_label = open("dev_label.txt").read().strip().split("\n") 
dev_label_int = []
for label in dev_label:
    dev_label_int.append(int(label))


# In[9]:


tst = open("tst_data.txt").read().strip().split("\n")


# In[10]:


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


# In[11]:


dataframe_words = pd.DataFrame(main_list)


# In[12]:


dataframe_words.columns= ["word"]


# In[13]:


count = dataframe_words.groupby('word')


# In[14]:


df=pd.DataFrame(count.size())
df.columns= ["frequency"]


# In[15]:


for index, row in df.iterrows():
    if row['frequency']<4 or row['frequency']>5000:
        df.drop(index, inplace=True)


# In[16]:


vector=[]
for index, row in df.iterrows():
       vector.append(index)
vector


# In[17]:


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


# In[18]:


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


# In[19]:


len(vector)


# In[20]:


import random
def pseudo_perceptron(X,Y):
    epochs = 5
    final_w=[]   
    df=pd.DataFrame(X)
    df['label']=Y
    df1 = df.values
   # print (len(w))
    for t in range(epochs):
        w=np.zeros(len(X[0])).tolist()
        np.random.shuffle(df1)
        X=df1[:,:-1]
        Y=df1[:,-1]
        for i in range(len(X)):
            a = 0
            row = X[i]
            for j in range(len(row)):
                a = a + row[j]*w[j]
            #a=(np.dot(X[i],w.transpose())).item(0)
            if(a>0):
                y_pred=1
            else:
                y_pred=0
            if(Y[i]!=y_pred):
                for j in range(len(row)):
                    w[j]=w[j]+row[j]*Y[i]-row[j]*y_pred
                    #w[j]=w[j]+s
        final_w.append(w)
    return final_w


# In[21]:


train_rate=pseudo_perceptron(train_data,trn_label_int)


# In[22]:


def test(X,W):
    y_pred=[]
    print (len(W))
    for t in range(len(W)):
        y_epoch=[]
        for i in range(len(X)):
            a=0
            row=X[i]
            for j in range(len(row)):
                a=a+row[j]*W[t][j]
            if a>0:
                y=1
            else:
                y=0
            y_epoch.append(y)
        y_pred.append(y_epoch)
    return y_pred


# In[23]:


train_pred=test(train_data,train_rate)


# In[24]:


dev_pred=test(dev_data,train_rate)


# In[25]:


from sklearn.metrics import accuracy_score
train_accuracy=[]
for i in range(len(train_pred)):
    train_accuracy.append(accuracy_score(trn_label_int, train_pred[i])*100)
dev_accuracy=[]
for i in range(len(dev_pred)):
    dev_accuracy.append(accuracy_score(dev_label_int, dev_pred[i])*100)


# In[26]:


train_accuracy


# In[28]:


from matplotlib import pyplot as plt
epoch=[1,2,3,4,5]
plt.ylabel('Accuracy Curve')
plt.plot(epoch, train_accuracy)
plt.plot(epoch, dev_accuracy)
plt.show()

