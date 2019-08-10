
# coding: utf-8

# read data

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


trn = open("trn.pos.txt").read().strip().split("\n")


# In[4]:


df_token = []
df_tag = []
tag_list1=[]
tag_list2=[]
for row in trn:
    row_split = row.split()
    tag_list1.append("START")
    for word in row_split:
        token,tag=word.split("/")
        df_token.append(token.lower())
        df_tag.append(tag)
        tag_list1.append(tag)
        tag_list2.append(tag)
    tag_list2.append("END")


# In[5]:


tag_list = list(zip(tag_list1,tag_list2))


# In[6]:


df=pd.DataFrame(df_token)


# In[7]:


df.columns= ["word"]


# In[8]:


count = df.groupby('word')


# In[9]:


df_t=pd.DataFrame(count.size())
df_t.columns= ["frequency"]


# In[10]:


df_unk=[]
df_final=[]
for index, row in df_t.iterrows():
    #print(index,row)
    if row['frequency']<6:
        df_unk.append(index)
    else:
        df_final.append(index)


# In[11]:


df_unk1=pd.DataFrame(df_unk)
df_final1=pd.DataFrame(df_final)


# In[12]:


df_final1.columns= ["word"]
df_unk1.columns= ["UNK"]


# In[13]:


index1=['A','C', 'D', 'M', 'N', 'O', 'P', 'R', 'V', 'W']
df_final.append('UNK')


# In[14]:


df_emission=pd.DataFrame(index=index1, columns=df_final)
df_emission_s=pd.DataFrame(index=index1, columns=df_final)


# In[15]:


df_num=pd.DataFrame(index=index1, columns=df_final)
df_denom=pd.DataFrame(index=index1, columns=df_final)


# In[16]:


df_num=df_num.fillna(0)
#df_denom=df_denom.fillna(0)


# In[17]:


index=['START','A','C', 'D', 'M', 'N', 'O', 'P', 'R', 'V', 'W']
column=['A','C', 'D', 'M', 'N', 'O', 'P', 'R', 'V', 'W','END']


# In[18]:


df_trans=pd.DataFrame(index=index, columns=column)
df_trans_s=pd.DataFrame(index=index, columns=column)


# In[19]:


beta=1
N=12
for index, row in df_trans.iterrows():
    for tag in column:
        count_i=0
        count_t=0
        for i in range(len(tag_list)):
            if tag_list[i][0]==index:
                count_i=count_i+1
                if tag_list[i][1]==tag:
                    count_t=count_t+1
        p=count_t/count_i
        p1=(count_t + beta)/(count_t + (N * beta))
        df_trans.loc[index,tag]=p
        df_trans_s.loc[index,tag]=p1


# In[20]:


j=0
for col in list(df_emission.columns):
    j+=1
    for i,k in enumerate(df_token):
        if col==df_token[i]:
            df_num.loc[df_tag[i],df_token[i]] += 1 
    if j%1000==0:
        print(j)
        print(col)
    


# In[21]:


j=0
for unk in list(df_unk):
    j+=1
    for i,k in enumerate(df_token):
        if unk==df_token[i]:
            df_num.loc[df_tag[i],'UNK'] += 1
    if j%1000==0:
        print(j)
        print(unk)


# In[22]:


import numpy as np
tag_corpus_list = np.array(df_tag)
index1=['A','C', 'D', 'M', 'N', 'O', 'P', 'R', 'V', 'W']
for tags in index1:
    tag_count = len(list(np.where(tag_corpus_list == tags)[0]))
    
    print(tag_count)
    for column in list(df_emission.columns):
        df_denom.loc[tags,column] = tag_count


# In[23]:


for tags in index1:
    #tag_count = len(list(np.where(tag_corpus_list == tags)[0]))
    
    #print(tag_count)
    for column in list(df_emission.columns):
        df_emission.loc[tags,column] = df_num.loc[tags,column]/df_denom.loc[tags,column]


# In[24]:


for tags in index1:
    #tag_count = len(list(np.where(tag_corpus_list == tags)[0]))
    
    #print(tag_count)
    #for column in list(df_emission.columns):
    df_emission.loc[tags,'UNK'] = df_num.loc[tags,'UNK']/df_denom.loc[tags,'UNK']


# In[25]:


alpha=1
V=len(df_final)
for tags in index1:
    #tag_count = len(list(np.where(tag_corpus_list == tags)[0]))
    
    #print(tag_count)
    for column in list(df_emission.columns):
        df_emission_s.loc[tags,column] = (df_num.loc[tags,column]+ alpha)/(df_denom.loc[tags,column]+ (V * alpha))


# In[26]:


for tags in index1:
    #tag_count = len(list(np.where(tag_corpus_list == tags)[0]))
    
    #print(tag_count)
    #for column in list(df_emission.columns):
    df_emission_s.loc[tags,'UNK'] = (df_num.loc[tags,'UNK']+ alpha)/(df_denom.loc[tags,'UNK']+ (V * alpha))


# In[27]:


#[computingID]-tprob.txt
df_trans.to_csv('og7ut-tprob.csv')
df_emission.to_csv('og7ut-eprob.csv')
df_trans_s.to_csv('og7ut-tprob-smoothed.csv')
df_emission_s.to_csv('og7ut-eprob-smoothed.csv')


# In[30]:


rows=['START','A','C', 'D', 'M', 'N', 'O', 'P', 'R', 'V', 'W']
columns=['A','C', 'D', 'M', 'N', 'O', 'P', 'R', 'V', 'W','END']
txt_file = "og7ut-tprob.txt"
with open(txt_file,"w") as output_file:
    for row in rows:
        for column in columns:
            parts = [row,column,str(df_trans.loc[row,column])]
            output_file.write(",".join(parts)+'\n')


# In[32]:


rows=['START','A','C', 'D', 'M', 'N', 'O', 'P', 'R', 'V', 'W']
columns=['A','C', 'D', 'M', 'N', 'O', 'P', 'R', 'V', 'W','END']
txt_file = "og7ut-tprob-smoothed.txt"
with open(txt_file,"w") as output_file:
    for row in rows:
        for column in columns:
            parts = [row,column,str(df_trans_s.loc[row,column])]
            output_file.write(",".join(parts)+'\n')


# In[31]:


rows=['A','C', 'D', 'M', 'N', 'O', 'P', 'R', 'V', 'W']
columns=df_final
txt_file = "og7ut-eprob.txt"
with open(txt_file,"w") as output_file:
    for row in rows:
        for column in columns:
            parts = [row,column,str(df_emission.loc[row,column])]
            output_file.write(",".join(parts)+'\n')


# In[33]:


rows=['A','C', 'D', 'M', 'N', 'O', 'P', 'R', 'V', 'W']
columns=df_final
txt_file = "og7ut-eprob-smoothed.txt"
with open(txt_file,"w") as output_file:
    for row in rows:
        for column in columns:
            parts = [row,column,str(df_emission_s.loc[row,column])]
            output_file.write(",".join(parts)+'\n')


# In[ ]:


df_emission_s


# In[34]:


dev = open("dev.pos.txt").read().strip().split("\n")


# In[37]:


def Viterbi(sentence,df_trans,df_emission,df_final):
    col_len=len(sentence)
    f=0
    tags=['A','C', 'D', 'M', 'N', 'O', 'P', 'R', 'V', 'W']
    numberofwords= list(range(1,col_len))
    sentencewithindex = []
    for i in range(len(sentence)):
        sentencewithindex.append(str(i)+sentence[i])
    df_v=pd.DataFrame(index=tags, columns=sentencewithindex)
    df_b=pd.DataFrame(index=tags, columns=numberofwords)
    #print(df_b)
    prev_word=[]
    y_list=[]
    for i,word in enumerate(sentence):
        if f==0:
            for k in index1:
                #if word in df_final :
                df_v.loc[k,sentencewithindex[i]]=df_trans.loc['START',k]+df_emission.loc[k,word]
            prev_word= sentencewithindex[i]
            f=1
        else:
            for k in index1:
                v=[]
               # print(len(v))
                for tag in tags:
                    #if word in df_final :
                    
                    t1 = df_v.loc[tag,prev_word]+df_trans.loc[tag,k]+df_emission.loc[k,word]
                    #print(df_v.loc[tag,prev_word])
                    #print(df_trans.loc[tag,k])
                    #print(df_emission.loc[k,word])
                    #print("done")
                    v.append(t1)
                    
                df_v.loc[k,sentencewithindex[i]]=np.amax(v)
                #print (t1)
                #print (np.argmax(v))
                df_b.loc[k,i]=tags[np.argmax(v)]
                v=[]
            prev_word= sentencewithindex[i]
        v=[]
    
    for tag in tags:
        t1=df_v.loc[tag,prev_word]+df_trans.loc[tag,'END']
        v.append(t1)
    v_end=np.amax(v)
    b_end=tags[np.argmax(v)]
    y_list.append(b_end)
    for i in range(col_len-1,1,-1):
        t=df_b.loc[b_end,i]
        b_end=t
        y_list.append(t)
    y_list=y_list[::-1]
    return y_list


# In[38]:


df_tag_dev = []
df_tag_pred=[]
i=0
for row in dev:
    i+=1
    if i%1000==0:
        print(i)
    #Viterbi(row,df_trans_s.apply(np.log),df_emission_s.apply(np.log))
    df_token_dev = []
    df_tag_d = []
    row_split = row.split()
    for word in row_split:
        token,tag=word.split("/")
       # print (token)
        if token in df_final:
            df_token_dev.append(token.lower())
        else:
            df_token_dev.append('UNK')
        df_tag_d.append(tag)
    #print(df_token_dev)
    value = Viterbi(df_token_dev,df_trans_s.applymap(np.log),df_emission_s.applymap(np.log),df_final)
    df_tag_pred.append(value)
    df_tag_dev.append(df_tag_d)
    #break


# In[ ]:


df_actual_tags = []
df_pred_tags = []
for lis in df_tag_pred:
    for val in lis:
        df_pred_tags.append(val)
        
        
for lis in df_tag_dev:
    for val in lis:
        df_actual_tags.append(val)
        
from sklearn.metrics import accuracy_score

print(accuracy_score(df_actual_tags, df_pred_tags)*100)


# In[39]:


tst = open("tst.word.txt").read().strip().split("\n")


# In[ ]:


df_tag_dev = []
df_tag_pred=[]
i=0
for row in tst:
    i+=1
    if i%1000==0:
        print(i)
    #Viterbi(row,df_trans_s.apply(np.log),df_emission_s.apply(np.log))
    df_token_dev = []
    #df_tag_d = []
    row_split = row.split()
    for word in row_split:
        token =word
       # print (token)
        if token in df_final:
            df_token_dev.append(token.lower())
        else:
            df_token_dev.append('UNK')
        #df_tag_d.append(tag)
    #print(df_token_dev)
    value = Viterbi(df_token_dev,df_trans_s.applymap(np.log),df_emission_s.applymap(np.log),df_final)
    df_tag_pred.append(value)
    #df_tag_dev.append(df_tag_d)
    #break


# In[ ]:


tst_data=[]
tst_final=[]
for i,row in enumerate(tst):
    row_split = row.split()
    tst_data=[]
    for j,word in enumerate(row_split):
        token= word + "/"+ df_tag_pred[i][j]
        tst_data.append(token)
    tst_sent=""
    for j,combo in enumerate(tst_data):
        if j==0:
            tst_sent=combo
        else:
            tst_sent=tst_sent+" "+combo
    tst_final.append(tst_sent)


# In[ ]:


txt_file = "og7ut-viterbi.txt"
with open(txt_file,"w") as output_file:
    for row in tst_final:
            output_file.write("\n".join(row))

