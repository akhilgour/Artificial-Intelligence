
# coding: utf-8

# 
# 1.  You are given the following playtennis table (S), that decides if a person wants to playtennis (target class) or not based on the features (outlook, temperature, humidity, wind)
# 
#            outlook       temperature       humidity        wind        playtennis
# 0           sunny            hot             high          weak            no
# 
# 1           sunny            hot             high          strong          no
# 
# 2           overcast         hot             high          weak            yes
# 
# 3           rain             mild            high          weak            yes
# 
# 4           rain             cool            normal        weak            yes
# 
# 5           rain             cool            normal        strong          no
# 
# 6           overcast         cool            normal        strong          yes
# 
# 7           sunny            mild            high          weak            no
# 
# 8           sunny            cool            normal        weak            yes
# 
# 9           rain             mild            normal        weak            yes
# 
# 10          sunny            mild            normal        strong          yes
# 
# 11          overcast         mild            high          strong          yes
# 
# 12          overcast         hot             normal        weak            yes
# 
# 13          rain             mild            high          strong          no
# 
# 14          sunny            mild            high          strong          no
# 
#     a. Compute the InformationGain(S, outlook), InformationGain(S, temperature), InformationGain(S, humidity), InformationGain(S, wind)
# 
#  
# 
#     b. Write a python code to build a decision stump which would be a decision tree of depth=1. The Decision Stump would be a class that would provide two primary functions: fit, and predict. The class should be able to handle discrete data that can be either numeric (eg, 0,1,2) or a string of characters (eg. Yes/No) . You can assume that all the features and the target are discrete valued attributes as shown in the “PlayTennis” dataset provided above. The single node in the tree must be chosen using information gain and entropy.  
# 
#      The fit method takes in two arguments: X is 2 dimensional numpy array, which is the training instances and Y which is the class labels corresponding to the training instances X.  Y will be a one dimensional numpy array. The fit method must take in the training data (X, Y) and build a decision stump.
# 
#      The predict method takes in a set of instances X_predict which has the same dimensions as the training instances X and will also be a 2-dimensional numpy array. The predict method must output a one dimensional array of the target classes predicted by the decision stump, corresponding to each of the X_predict instances.
# 
#      You can test your code with the same “playtennis” dataset provided above to see if it computes the correct information gain for each of the features of the “playtennis” data. Please make sure your code also handles boundary test conditions such as an empty training dataset.
# 
#      Please name your class as “DecisionStump” and the two methods as “fit” and “predict”.
# 
#     Note: The decision stump can be used to select a single important feature that helps predict the target class, given a large set of features.
# 
#  
# 
# 2. Load the Boston housing dataset
# 
# from sklearn.datasets import load_boston
# 
# boston = load_boston()
# 
# Build your train and test data by partitioning into 70% train and 30% test. Since this data is predicting the price of the housing market, the output is real-valued.  You will be using decision trees and boosting algorithms for predicting a continuous output data. Hence, you will use the DecisionTreeRegressor and the AdaBoostRegressor algorithm in Python for predicting the housing price.  In addition, you can compute the “Mean squared Error” to determine the error on your dataset from the sklearn.metrics library. Modify the parameters of the Boosting and Decision Tree algorithm and see if it can improve your performance. Set the random_state to 1 when you are doing the train-test-split and in all of the algorithms you use. Please provide the code you used. Provide a summary of the different parameters you tried and the mean squared error you obtained for the boosting algorithm. Please, also report the best mean-squared error you obtained on your test dataset.

# ## Assignment 1
# ### Akhil Gour
# ### 012455586
# 
# 
# #### Solution 1:
# To calculate the entropy of the whole Play tennis datasets (S):
# 
# 9 out of 15 instances are “Yes” and 6 out of 15 instances are “No”.
# 
# P(yes) =  -(9/15) log(9/15) = -0.6*(-0.7369655941662062) = -0.6*-0.74 = 0.44
# P(no) = - (6/15) log(6/15) = -(0.4)*(-1.3219280948873622) = -0.4*-1.32 = 0.53
# 
# H(S) = P(yes) + P(no)  = 0.44 + 0.53 = 0.97
# 
# 
# Information Gain(S, outlook):
# E(Outlook = ‘‘Sunny’’) = -(2/6)log(2/6) - (4/6)log(4/6) = -(0.33)*(-1.6) -(0.66)*(-0.6) = 0.53 + 0.4 = 0.93
# E(Outlook = ‘Overcast’) = -1*log(1) -(0)*log(0) = 0
# E(Outlook = ‘Rain’) =  -(3/5)*log(3/5) -(2/5)*log(2/5) = 0.97
# 
# Average Entropy of Outlook = 6/15*(0.93) + 4*0 + 5/15*0.97 = 0.37 + 0.32 = 0.69
# 
# Information Gain(S, outlook) = 
# = H(S) - (E(Outlook = ‘Sunny’) + E(Outlook = ‘Overcast’) + E(Outlook = ‘Rain’) ) 
# = 0.97 - 0.69 = 0.28
# 
# Information Gain(S, Temperature):
# 
# E(Temperature = ‘Hot’) = -(2/4)log(2/4) - (2/4)log(2/4) = -(0.5)*(-1) -(0.5)*(-1) = 1
# 
# E(Temperature = ‘Mild’) = -(4/7)*log(4/7) -(3/7)*log(3/7) = -(0.57)*(-0.8) -(0.43)*(-1.22) = 0.46 + 0.52 = 0.98 
# 
# E(Temperature = ‘Cool’) =  -(1/4)*log(1/4) -(3/4)*log(3/4) = -0.25*(-2) - 0.75*(-0.4)= 0.5+0.3 = 0.8
# 
# Average Entropy of Temperature= 4/15*1+ 7/15*0.98 + 4/15*0.8 = 0.27 + 0.46 + 0.2 = 0.94
# 
# Information Gain(S, Temperature) 
# = H(S) - (Average Entropy of Temperature) 
# = 0.97 - 0.94 = 0.03
# 
# 
# Information Gain(S, ‘Humidity’):
# E(‘Humidity’ = High) 
# = -(3/8)log(3/8) - (5/8)log(5/8) = -(0.375)*(-1.4) -(0.625)*(-0.68) = 0.525 + 0.425 = 0.95
# E(‘Humidity’ = Normal) 
# = -(6/7)*log(6/7) -(1/7)*log(1/7) = -(0.86)*(-0.22) -(0.14)*(-2.84) = 0.19 + 0.41 = 0.60
# 
# Average Entropy of ‘Humidity’= 8/15*0.95+ 7/15*0.60 = 0.51 + 0.28 = 0.79
# 
# Information Gain(S, ‘Humidity’) 
# = H(S) - (Average Entropy of ‘Humidity’) 
# = 0.97 - 0.79 = 0.18
# 
# Information Gain(S, ‘Wind’):
# E(‘Wind’ = Weak) = -(6/8)log(6/8) - (2/8)log(2/8) = -(0.75)*(-0.4) -(0.25)*(-2) = 0.3 + 0.5 = 0.8
# E(‘Wind’ = Strong) = -(4/7)*log(4/7) -(3/7)*log(3/7) = -(0.57)*(-0.8) -(0.43)*(-1.22) = 0.46 + 0.52 = 0.98 
# 
# Average Entropy of ‘Wind’= 8/15*0.8+ 7/15*0.98= 0.43 + 0.46 = 0.89
# 
# Information Gain(S, ‘Wind’) 
# = H(S) - (Average Entropy of ‘Wind’) 
# = 0.97 - 0.89 = 0.08
# 
# 
# 

# ## Solution2 ##

# In[1]:


import numpy as np
import pandas as pd
import math
from collections import Counter
import operator
from pprint import pprint
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


X = np.array([['sunny','hot','high','weak'],
        ['sunny','hot','high','strong'],
        ['overcast','hot','high','weak'],
        ['rain','mild','high','weak'],
        ['rain','cool','normal','weak'],
        ['rain','cool','normal','strong'],
        ['overcast','cool','normal','strong'],
        ['sunny','mild','high','weak'],
        ['sunny','cool','normal','weak'],
        ['rain','mild','normal','weak'],
        ['sunny','mild','normal','strong'],
        ['overcast','mild','high','strong'],
        ['overcast','hot','normal','weak'],
        ['rain','mild','high','strong'],
        ['sunny','mild','high','strong']
    ])

Y = np.array(['no','no','yes','yes','yes','no','yes','no','yes','yes','yes','yes','yes','no','no'])


# In[3]:


class DStump:
    def __init__(self):
        pass
    ct = defaultdict(list)
    def entropy(self,s):
        res = 0
        val, counts = np.unique(s, return_counts=True)
        freqs = counts.astype('float')/len(s)
        for p in freqs:
            if p != 0.0:
                res -= p * np.log2(p)
        return res
    def info_gain(self,y, x):
        res = self.entropy(y)
        val, counts = np.unique(x, return_counts=True)
        freqs = counts.astype('float')/len(x)
        for p, v in zip(freqs, val):
            res -= p * self.entropy(y[x == v])
        return res
    def partition(self,a):
        return {c: (a==c).nonzero()[0] for c in np.unique(a)}
    def fit(self,x,y):
        self.Y = Y
        if(len(set(Y)) == 1):
            return y
        gain = np.array([self.info_gain(Y, x_attr) for x_attr in X.T])
        self.feature = np.argmax(gain)
        self.sets = self.partition(X[:, self.feature])
    ct = defaultdict(list)
    def predict(self,testdata):
        result1 = []
        c = testdata[:,self.feature]
        for i in range(0,len(c)):
            val = self.sets.get(c[i])
            y_subset = self.Y.take(val, axis=0)
            target = Counter(y_subset)
            result1.append(max(target.items(), key=operator.itemgetter(1))[0])
        return result1


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=1)


# In[5]:


Dstump = DStump()


# In[6]:


Dstump.fit(X_train,y_train)


# In[7]:


y_pred = Dstump.predict(X_test)


# In[8]:


print('Accuracy: %.2f',accuracy_score(y_test, y_pred))


# ## Solution 3

# In[9]:


from sklearn.datasets import load_boston

boston = load_boston()


# In[10]:


from sklearn.model_selection import train_test_split 


# In[11]:


boston


# In[12]:


import pandas as pd


# In[13]:


X = pd.DataFrame(boston.data)


# In[14]:


X.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
        'TAX', 'PTRATIO', 'B', 'LSTAT']


# In[15]:


X


# In[16]:


Y = pd.DataFrame(boston.target)


# In[17]:


Y


# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=1)


# In[20]:


from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor


# In[21]:


from sklearn.metrics import mean_absolute_error


# In[22]:


DTregressor = DecisionTreeRegressor(random_state=1)


# In[23]:


DTregressor.fit(X_train,Y_train)


# In[24]:


from sklearn.metrics import mean_squared_error


# In[25]:


Y_pred_DTregressor = DTregressor.predict(X_test)


# In[26]:


mse = mean_squared_error(Y_test, Y_pred_DTregressor)


# In[27]:


print('Mean Squared error before boosting',mse)


# In[28]:


import warnings
warnings.filterwarnings('ignore')


# In[29]:


for i in range(1,30):
    adaboost = AdaBoostRegressor(base_estimator=DTregressor, n_estimators=i, random_state=1)
    adaboost.fit(X_train,Y_train)
    Y_pred_boost = adaboost.predict(X_test)
    print('Mean Squared Error: %.2f' %mean_squared_error(Y_test, Y_pred_boost))


# In[30]:


adaboost = AdaBoostRegressor(base_estimator=DTregressor, n_estimators=10, random_state=1)
adaboost.fit(X_train,Y_train)
Y_pred_boost = adaboost.predict(X_test)
print('Mean Squared Error after boosting: %.2f' %mean_squared_error(Y_test, Y_pred_boost))

