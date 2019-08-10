
# coding: utf-8

# read data

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


# In[2]:


trn = open("trn_data.txt").read().strip().split("\n")
trn_label = open("trn_label.txt").read().strip().split("\n")
trn_label_int = []
for label in trn_label:
    trn_label_int.append(int(label))


# In[3]:


trn


# In[4]:


trn_label_int


# In[5]:


dev = open("dev_data.txt").read().strip().split("\n")
dev_label = open("dev_label.txt").read().strip().split("\n") 


# In[6]:


dev_label_int = []
for label in dev_label:
    dev_label_int.append(int(label))


# In[7]:


tst = open("tst_data.txt").read().strip().split("\n")


# Preprocessing the training data

# In[8]:


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


# In[9]:


main_list


# In[10]:


dataframe_words = pd.DataFrame(main_list)


# In[11]:


dataframe_words.columns= ["word"]


# In[12]:


dataframe_words.head()


# In[13]:


count = dataframe_words.groupby('word')


# In[14]:


print (count.size())


# In[15]:


df=pd.DataFrame(count.size())
df.columns= ["frequency"]


# In[16]:


df.head()


# In[17]:


for index, row in df.iterrows():
    if row['frequency']<4 or row['frequency']>5000:
        df.drop(index, inplace=True)


# In[18]:


df.head()


# In[19]:


vector=[]
for index, row in df.iterrows():
       vector.append(index)
vector


# In[20]:


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


# In[21]:


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


# In[22]:


len(vector)


# In[62]:


df2=pd.DataFrame(train_data)
df2['label']=trn_label_int


# In[63]:


df3 = df2.values
np.random.shuffle(df3)
X=df3[:,:-1]
Y=df3[:,-1]


# In[64]:


X


# In[65]:


Y


# In[79]:


np.random.shuffle(df3)
X=df3[:,:-1]
Y=df3[:,-1]


# In[60]:


X


# In[80]:


Y


# In[68]:


len(dev_data)


# In[67]:


train_data


# In[83]:


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


# In[84]:


train_rate=pseudo_perceptron(train_data,trn_label_int)


# In[85]:


train_rate


# In[95]:


len(train_rate)


# In[87]:


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


# In[88]:


train_pred=test(train_data,train_rate)


# In[89]:


dev_pred=test(dev_data,train_rate)


# In[90]:


len(dev_pred[0])


# In[91]:


from sklearn.metrics import accuracy_score
train_accuracy=[]
for i in range(len(train_pred)):
    train_accuracy.append(accuracy_score(trn_label_int, train_pred[i])*100)
dev_accuracy=[]
for i in range(len(dev_pred)):
    dev_accuracy.append(accuracy_score(dev_label_int, dev_pred[i])*100)


# In[93]:


train_accuracy


# In[94]:


from matplotlib import pyplot as plt
epoch=[1,2,3,4,5]
plt.ylabel('Accuracy Curve')
plt.plot(epoch, train_accuracy)
plt.plot(epoch, dev_accuracy)
plt.show()


# In[102]:


def avg_perceptron(X,Y):
    epochs = 5
    final_w=[]   
    df=pd.DataFrame(X)
    df['label']=Y
    df1 = df.values
    n=len(X)
    w=np.zeros(len(X[0])).tolist()
    for t in range(epochs):
        np.random.shuffle(df1)
        X=df1[:,:-1]
        Y=df1[:,-1]
        sum1=np.zeros(len(w)).tolist()
        avg=np.zeros(len(w)).tolist()
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
                    sum1[j]=sum1[j]+w[j]
            else:
                for j in range(len(row)):
                    sum1[j]=sum1[j]+w[j]                
        for i in range(len(w)):
            avg[i]=sum1[i]/n
        w=avg
        final_w.append(w)
    return final_w


# In[103]:


train_rate_avg=avg_perceptron(train_data,trn_label_int)


# In[105]:


train_avg_pred=test(train_data,train_rate_avg)


# In[111]:


dev_avg_pred=test(dev_data,train_rate_avg)


# In[112]:


len(dev_avg_pred[0])


# In[113]:


train_avg_accuracy=[]
for i in range(len(train_avg_pred)):
    train_avg_accuracy.append(accuracy_score(trn_label_int, train_avg_pred[i])*100)
dev_avg_accuracy=[]
for i in range(len(dev_avg_pred)):
    dev_avg_accuracy.append(accuracy_score(dev_label_int, dev_avg_pred[i])*100)


# In[114]:


train_avg_accuracy


# In[115]:


dev_avg_accuracy


# In[126]:


epoch=[1,2,3,4,5]
plt.ylabel('Accuracy Curve')
plt.plot(epoch, train_avg_accuracy)
plt.plot(epoch, dev_avg_accuracy)
plt.show()


# In[118]:


test_data=[]
a=1
b=0
for review in tst:
    review_split = review.split()
    rows=[]
    for word in vector:
        if word in review_split:
            rows.append(a)
        else:
            rows.append(b)
    test_data.append(rows) 


# In[123]:


test_avg_pred=test(test_data,train_rate_avg)


# In[125]:


test_avg_pred[4]

