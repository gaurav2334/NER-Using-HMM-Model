#!/usr/bin/env python
# coding: utf-8

# In[180]:


with open (r"D:\M.Tech 2nd sem\NLP\Natural Language Processing\Assignments\Assignment_02\NER-Dataset-Train.txt") as f:
    lines=f.readlines()


# In[181]:


lines[0]
pairs=[[]]
for i in lines:
    if i=='\n':
        pairs.append([])
        continue
    strip=i.strip('\n').split('\t')
    pairs[-1].append(strip)


# In[182]:


UniqueTags=set()
UniqueWords=set()
for i in pairs:
    for ind,[j,k] in enumerate(i):
        UniqueTags.add(k)
        UniqueWords.add(j)


# In[183]:


pairs=[x for x in pairs if len(x)>3]


# In[184]:


len(pairs)


# In[185]:


UniqueWords,UniqueTags=list(UniqueWords),list(UniqueTags)
len(UniqueWords),len(UniqueTags)


# In[186]:


UniqueTags_to_numbers={i:ind for ind,i in enumerate(UniqueTags)}
UniqueWords_to_numbers={i:ind for ind,i in enumerate(UniqueWords)}
numbers_to_tags={j:i for i,j in UniqueTags_to_numbers.items()}


# In[195]:


import numpy as np
y_true=[]
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
import random
class HMM:
    def __init__(self,value):
        self.value=value
        self.pi=np.ones((3),dtype='float64') # pi initial probabilities
        self.TMatrix=np.ones((3,3),dtype='float64') # Transision matrix for bigram
        self.EMatrix=np.ones((3,len(UniqueWords)),dtype='float64') # Emmision matrix
        self.train_data=None
    def train(self,train_data):     
        previous=None #Three pointers used to keep track of the pre  , current and next word
        next=None
        for i in train_data:
            for ind,[j,k] in enumerate(i):
                if ind==0:
                    self.pi[UniqueTags_to_numbers[k]]+=1
                    previous=k
                    continue
                next=k
                self.TMatrix[UniqueTags_to_numbers[previous]][UniqueTags_to_numbers[next]]+=1
                previous=next
                self.EMatrix[UniqueTags_to_numbers[k]][UniqueWords_to_numbers[j]]+=1
        self.pi=self.pi/pi.sum()
        self.TMatrix=self.TMatrix/self.TMatrix.sum(axis=1)[:,np.newaxis]
        self.EMatrix=self.EMatrix/self.EMatrix.sum(axis=1)[:,np.newaxis]
    def viterbi(self,words):
        for j in words:
            if j not in UniqueWords_to_numbers:
                UniqueWords_to_numbers[j]=1
        matrix=np.zeros((len(words),3))-1e+4
        for i in range(len(words)):
            if i==0:
                matrix[i]=self.pi*self.EMatrix[:,UniqueWords_to_numbers[words[i]]]
                matrix[i]=np.log(matrix[i])
                continue
            for j in range(3):
                for k in range(3):
                    matrix[i][j]=max(matrix[i-1][k]+np.log(self.TMatrix[k][j])*                                     self.value+np.log(self.EMatrix[j][UniqueWords_to_numbers[words[i]]]),
                                                matrix[i][j] )
        resultseq=[]
        for i in np.argmax(matrix,axis=1):
            resultseq.append(numbers_to_tags[i])
        return resultseq
    def test(self,test_data):
        return_sequences=[]
        y_pred=[]
        y_true=[]
        for i in test_data:
            i=np.array(i)
            # print(i)
            y_true+=[x for x in i[:,1]]
            y_pred+=self.viterbi(i[:,0])
        return precision_recall_fscore_support(y_pred,y_true,average='macro',zero_division=0)
    def FiveFold(self,data):
        index=int(len(data)/5)
        _=["Unigram Model","Bigram Model"]
        print(f"{'*'*60}\nFor {_[self.value]}\n") 
        for i in range(5):
            train=data[0:index*i]+data[index*(i+1):]
            test=data[index*i:index*(i+1)]
            self.train(train)
            test_result=self.test(test)
            print(f"Precision for {i+1}th fold=",round(test_result[0],2),
                  f"\tRecall for {i+1}th fold=",round(test_result[1],2),
                  f"\tF1_score for {i+1}th fold=",round(test_result[2],2),"\n")
            self.__init__(self.value)        


# In[196]:


hmm=HMM(1)
hmm.FiveFold(pairs)
hmm=HMM(0)
hmm.FiveFold(pairs)

