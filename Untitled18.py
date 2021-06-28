#!/usr/bin/env python
# coding: utf-8

# In[ ]:


Users(/shadabahmed/Downloads/kaggle, nlp/train.csv)


# In[89]:


import pandas as pd
import numpy as np
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
import os
os.getcwd()
train=pd.read_csv("/Users/shadabahmed/Downloads/kaggle nlp/train.csv")
test=pd.read_csv("/Users/shadabahmed/Downloads/kaggle nlp/test.csv")
print("The number of sample in the training data is {}".format(train.shape[0]))
print("The number of sample in the training data is {}".format(train.shape[1]))
print("The number of sample in the test data is {}".format(test.shape[0]))
print("The number of sample in the test data is {}".format(test.shape[1]))

print(train.isnull().sum())
print(test.isnull().sum())


# In[90]:


import matplotlib.pyplot as plt
#length of text in train and test data
length_train=train['text'].str.len()
length_test=test['text'].str.len()
plt.hist(length_train,bins=20,label="train_text")
plt.hist(length_test,bins=20,label="test_text")
plt.legend()
plt.show()


# In[91]:


train['check']='train'
test['check']='test'
combined=train.append(test,ignore_index=True)


# In[92]:


combined.tail()


# In[60]:


def remove_pattern(input_txt,pattern):
    r=re.findall(pattern,input_txt)
    for i in r:
        input_txt=re.sub(i,'',input_txt)
    return input_txt


# In[93]:


combined['neat_text']=np.vectorize(remove_pattern)(combined['text'],"@[\w*]")
combined['neat_text']=combined['neat_text'].str.replace('[^a-zA-Z#]'," ")
combined['neat_text']=combined['neat_text'].apply(lambda x:x.lower())
combined['neat_text']=combined['neat_text'].apply(lambda x: " ".join([w for w in x.split() if len(w)>3]))
combined['neat_text']=combined['neat_text'].apply(lambda x:x.split())


# In[94]:


from nltk.stem import WordNetLemmatizer
wordnet=WordNetLemmatizer()
combined['neat_text']=combined['neat_text'].apply(lambda x:[wordnet.lemmatize(i) for i in x])


# In[95]:


combined.head()


# In[96]:


#stiching the tweets together by nltk moses detokenizer
for i in range(len(combined['neat_text'])):
    combined['neat_text'][i]=' '.join(combined['neat_text'][i])
    


# In[97]:


combined.head()


# In[108]:


all_words=' '.join([text for text in combined['neat_text']])
from wordcloud import WordCloud
wordcloud=WordCloud(width=800,height=500,random_state=21,max_font_size=80).generate(all_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud,interpolation="bilinear")
plt.axis('off')
plt.show()


# In[109]:


normal_words=' '.join([text for text in combined['neat_text'][combined['target']==0]])
from wordcloud import WordCloud
wordcloud=WordCloud(width=800,height=500,random_state=21,max_font_size=80).generate(normal_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud,interpolation="bilinear")
plt.axis('off')
plt.show()


# In[110]:


negative_words=' '.join([text for text in combined['neat_text'][combined['target']==1]])
from wordcloud import WordCloud
wordcloud=WordCloud(width=800,height=500,random_state=21,max_font_size=80).generate(negative_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud,interpolation="bilinear")
plt.axis('off')
plt.show()


# In[111]:


combined.head()


# In[112]:


def hashtag_extract(x):
    hashtags=[]
    for i in x:
        ht=re.findall(r"#(\w+)",i)
        hashtags.append(ht)
    return hashtags


# In[125]:


HT_regular=hashtag_extract(combined['neat_text'][combined['target']==0.0])
HT_destructive=hashtag_extract(combined['neat_text'][combined['target']==1.0])
HT_regular=sum(HT_regular,[])
HT_destructive=sum(HT_destructive,[])


# In[126]:


a=nltk.FreqDist(HT_regular)
d=pd.DataFrame({'Hashtag':list(a.keys()),'Count':list(a.values())})


# In[140]:


import seaborn as sns
d=d.nlargest(columns="Count",n=20)
plt.figure(figsize=(10,5))
ax=sns.barplot(data=d,x="Hashtag",y="Count")
ax.set(ylabel='Count')
plt.show()


# In[143]:


b=nltk.FreqDist(HT_destructive)
e=pd.DataFrame({'Hashtag':list(b.keys()),'Count':list(b.values())})
import seaborn as sns
e=e.nlargest(columns="Count",n=10)
plt.figure(figsize=(10,5))
ax=sns.barplot(data=e,x="Hashtag",y="Count")
ax.set(ylabel='Count')
plt.show()


# In[129]:


HT_destructive


# In[130]:


HT_regular


# In[ ]:




