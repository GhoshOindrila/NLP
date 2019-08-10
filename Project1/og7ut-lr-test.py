
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression as LR


# In[2]:


trn = open("trn_data.txt").read().strip().split("\n")
trn_label = open("trn_label.txt").read().strip().split("\n")
trn_label_int = []
for label in trn_label:
    trn_label_int.append(int(label))


# In[3]:


dev = open("dev_data.txt").read().strip().split("\n")
dev_label = open("dev_label.txt").read().strip().split("\n") 


# In[4]:


dev_label_int = []
for label in dev_label:
    dev_label_int.append(int(label))


# In[5]:


tst = open("tst_data.txt").read().strip().split("\n")


# In[6]:


vectorizer = CountVectorizer()


# In[8]:


train_data = vectorizer.fit_transform(trn)
print (train_data.shape)


# In[9]:


dev_data = vectorizer.transform(dev)
print (dev_data.shape)
test_data = vectorizer.transform(tst)
print (test_data.shape)


# In[11]:


classifier = LR()
classifier.fit(train_data, trn_label_int)


# In[12]:


Train_accuracy =classifier.score(train_data, trn_label_int)
Dev_accuracy =classifier.score(dev_data, dev_label_int)


# In[19]:


Train_accuracy * 100


# In[20]:


Dev_accuracy * 100


# In[15]:


vectorizer1 = CountVectorizer(ngram_range=(1,2))


# In[16]:


train_data1 = vectorizer1.fit_transform(trn)
print (train_data1.shape)


# In[18]:


dev_data1 = vectorizer1.transform(dev)
print (dev_data1.shape)
test_data1 = vectorizer1.transform(tst)
print (test_data1.shape)


# In[21]:


classifier.fit(train_data1, trn_label_int)


# In[22]:


Train_accuracy1 =classifier.score(train_data1, trn_label_int)
Dev_accuracy1 =classifier.score(dev_data1, dev_label_int)


# In[23]:


Train_accuracy1 * 100


# In[24]:


Dev_accuracy1 * 100


# In[26]:


lamda=[0.0001,0.001,0.01,0.1,1,10,100]
c=np.zeros(len(lamda)).tolist()
for i in range(len(lamda)):
    c[i]=1/lamda[i]


# In[27]:


classifier_list=np.zeros(len(c)).tolist()
for i in range(len(c)):
    classifier_list[i] = LR(C=c[i])


# In[28]:


Train_accuracy_list=np.zeros(len(c)).tolist()
Dev_accuracy_list=np.zeros(len(c)).tolist()
for i in range(len(c)):
    classifier_list[i].fit(train_data1, trn_label_int)
    Train_accuracy_list[i] =classifier_list[i].score(train_data1, trn_label_int)
    Dev_accuracy_list[i] =classifier_list[i].score(dev_data1, dev_label_int)


# In[32]:


for i in range(len(c)):
    print (Train_accuracy_list[i]*100)
    print (Dev_accuracy_list[i]*100)


# In[37]:


lamda_rev=[0.01,0.03,0.06,0.09,0.1,0.5,1]
c1=np.zeros(len(lamda_rev)).tolist()
for i in range(len(lamda_rev)):
    c1[i]=1/lamda_rev[i]


# In[38]:


classifier_list_rev=np.zeros(len(c1)).tolist()
for i in range(len(c1)):
    classifier_list_rev[i] = LR(C=c1[i])


# In[39]:


Train_accuracy_list_rev=np.zeros(len(c1)).tolist()
Dev_accuracy_list_rev=np.zeros(len(c1)).tolist()
for i in range(len(c1)):
    classifier_list_rev[i].fit(train_data1, trn_label_int)
    Train_accuracy_list_rev[i] =classifier_list_rev[i].score(train_data1, trn_label_int)
    Dev_accuracy_list_rev[i] =classifier_list_rev[i].score(dev_data1, dev_label_int)


# In[41]:


for i in range(len(c1)):
    print ("Train-> ",Train_accuracy_list_rev[i]*100)
    print ("Dev-> ",Dev_accuracy_list_rev[i]*100)


# In[42]:


lamda_rev1=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
c2=np.zeros(len(lamda_rev1)).tolist()
for i in range(len(lamda_rev1)):
    c2[i]=1/lamda_rev1[i]


# In[43]:


classifier_list_rev1=np.zeros(len(c2)).tolist()
for i in range(len(c2)):
    classifier_list_rev1[i] = LR(C=c2[i])


# In[44]:


Train_accuracy_list_rev1=np.zeros(len(c2)).tolist()
Dev_accuracy_list_rev1=np.zeros(len(c2)).tolist()
for i in range(len(c2)):
    classifier_list_rev1[i].fit(train_data1, trn_label_int)
    Train_accuracy_list_rev1[i] =classifier_list_rev1[i].score(train_data1, trn_label_int)
    Dev_accuracy_list_rev1[i] =classifier_list_rev1[i].score(dev_data1, dev_label_int)


# In[46]:


for i in range(len(c2)):
    print ("Train-> ",Train_accuracy_list_rev1[i]*100)
    print ("Dev-> ",Dev_accuracy_list_rev1[i]*100)


# In[47]:


lamda_rev2=[0.45,0.55,0.85,0.95,1.1]
c3=np.zeros(len(lamda_rev2)).tolist()
for i in range(len(lamda_rev2)):
    c3[i]=1/lamda_rev2[i]


# In[48]:


classifier_list_rev2=np.zeros(len(c3)).tolist()
for i in range(len(c3)):
    classifier_list_rev2[i] = LR(C=c3[i])


# In[49]:


Train_accuracy_list_rev2=np.zeros(len(c3)).tolist()
Dev_accuracy_list_rev2=np.zeros(len(c3)).tolist()
for i in range(len(c3)):
    classifier_list_rev2[i].fit(train_data1, trn_label_int)
    Train_accuracy_list_rev2[i] =classifier_list_rev2[i].score(train_data1, trn_label_int)
    Dev_accuracy_list_rev2[i] =classifier_list_rev2[i].score(dev_data1, dev_label_int)


# In[50]:


for i in range(len(c3)):
    print ("Train-> ",Train_accuracy_list_rev2[i]*100)
    print ("Dev-> ",Dev_accuracy_list_rev2[i]*100)


# In[51]:


lamda_rev3=[0.88,0.92,0.96,1.01]
c4=np.zeros(len(lamda_rev3)).tolist()
for i in range(len(lamda_rev3)):
    c4[i]=1/lamda_rev3[i]


# In[52]:


classifier_list_rev3=np.zeros(len(c4)).tolist()
for i in range(len(c4)):
    classifier_list_rev3[i] = LR(C=c4[i])


# In[53]:


Train_accuracy_list_rev3=np.zeros(len(c4)).tolist()
Dev_accuracy_list_rev3=np.zeros(len(c4)).tolist()
for i in range(len(c4)):
    classifier_list_rev3[i].fit(train_data1, trn_label_int)
    Train_accuracy_list_rev3[i] =classifier_list_rev3[i].score(train_data1, trn_label_int)
    Dev_accuracy_list_rev3[i] =classifier_list_rev3[i].score(dev_data1, dev_label_int)


# In[54]:


for i in range(len(c4)):
    print ("Train-> ",Train_accuracy_list_rev3[i]*100)
    print ("Dev-> ",Dev_accuracy_list_rev3[i]*100)


# In[56]:


lamda_rev4=[0.96,0.975,0.98,0.985,0.9,0.99]
c5=np.zeros(len(lamda_rev4)).tolist()
for i in range(len(lamda_rev4)):
    c5[i]=1/lamda_rev4[i]


# In[57]:


classifier_list_rev4=np.zeros(len(c5)).tolist()
for i in range(len(c5)):
    classifier_list_rev4[i] = LR(C=c5[i])


# In[58]:


Train_accuracy_list_rev4=np.zeros(len(c5)).tolist()
Dev_accuracy_list_rev4=np.zeros(len(c5)).tolist()
for i in range(len(c5)):
    classifier_list_rev4[i].fit(train_data1, trn_label_int)
    Train_accuracy_list_rev4[i] =classifier_list_rev4[i].score(train_data1, trn_label_int)
    Dev_accuracy_list_rev4[i] =classifier_list_rev4[i].score(dev_data1, dev_label_int)


# In[59]:


for i in range(len(c5)):
    print ("Train-> ",Train_accuracy_list_rev4[i]*100)
    print ("Dev-> ",Dev_accuracy_list_rev4[i]*100)


# In[60]:


lamda_rev5=[0.961,0.963,0.965,0.967,0.969,0.971,0.973]
c6=np.zeros(len(lamda_rev5)).tolist()
for i in range(len(lamda_rev5)):
    c6[i]=1/lamda_rev5[i]


# In[61]:


classifier_list_rev5=np.zeros(len(c6)).tolist()
for i in range(len(c6)):
    classifier_list_rev5[i] = LR(C=c6[i])


# In[62]:


Train_accuracy_list_rev5=np.zeros(len(c6)).tolist()
Dev_accuracy_list_rev5=np.zeros(len(c6)).tolist()
for i in range(len(c6)):
    classifier_list_rev5[i].fit(train_data1, trn_label_int)
    Train_accuracy_list_rev5[i] =classifier_list_rev5[i].score(train_data1, trn_label_int)
    Dev_accuracy_list_rev5[i] =classifier_list_rev5[i].score(dev_data1, dev_label_int)


# In[63]:


for i in range(len(c6)):
    print ("Train-> ",Train_accuracy_list_rev5[i]*100)
    print ("Dev-> ",Dev_accuracy_list_rev5[i]*100)


# In[64]:


final_lamda=[0.963,0.965,0.967,0.968]
c_final=np.zeros(len(final_lamda)).tolist()
for i in range(len(final_lamda)):
    c_final[i]=1/final_lamda[i]


# In[66]:


classifier_final=np.zeros(len(c_final)).tolist()
for i in range(len(c_final)):
    classifier_final[i] = LR(C=c_final[i],penalty='l1')


# In[67]:


Train_accuracy_final=np.zeros(len(c_final)).tolist()
Dev_accuracy_final=np.zeros(len(c_final)).tolist()
for i in range(len(c_final)):
    classifier_final[i].fit(train_data1, trn_label_int)
    Train_accuracy_final[i] =classifier_final[i].score(train_data1, trn_label_int)
    Dev_accuracy_final[i] =classifier_final[i].score(dev_data1, dev_label_int)


# In[68]:


for i in range(len(c_final)):
    print ("Train-> ",Train_accuracy_final[i]*100)
    print ("Dev-> ",Dev_accuracy_final[i]*100)


# In[118]:


vectorizer5=CountVectorizer(lowercase=True, min_df=2,ngram_range=(1,3), stop_words='english')


# In[119]:


train_data5 = vectorizer5.fit_transform(trn)
print (train_data5.shape)


# In[120]:


dev_data5 = vectorizer5.transform(dev)
print (dev_data5.shape)
test_data5 = vectorizer5.transform(tst)
print (test_data5.shape)


# In[137]:


lamda5=[0.94,0.95,0.968,1,8.5]
c_list5=np.zeros(len(lamda5)).tolist()
for i in range(len(lamda5)):
    c_list5[i]=1/lamda5[i]


# In[138]:


classifier5=np.zeros(len(c_list5)).tolist()
for i in range(len(c_list5)):
    classifier5[i] = LR(C=c_list5[i],solver="lbfgs")


# In[139]:


Train_accuracy5=np.zeros(len(c_list5)).tolist()
Dev_accuracy5=np.zeros(len(c_list5)).tolist()
for i in range(len(c_list5)):
    classifier5[i].fit(train_data5, trn_label_int)
    Train_accuracy5[i] =classifier5[i].score(train_data5, trn_label_int)
    Dev_accuracy5[i] =classifier5[i].score(dev_data5, dev_label_int)


# In[140]:


for i in range(len(c_list5)):
    print ("Train-> ",Train_accuracy5[i]*100)
    print ("Dev-> ",Dev_accuracy5[i]*100)


# In[141]:


Test_predict=np.zeros(len(c_list5)).tolist()
for i in range(len(c_list5)):
    Test_predict[i]=classifier5[i].predict(test_data5)


# In[142]:


Test_predict

