
# coding: utf-8

# Akhil Kumar Gour
# SJSU ID: 012455586

# Simple assignment to learn how to use Python Scikit-learn library to build a decision tree using a sample dataset.

# In[ ]:


import pandas as pd


# In[27]:


from sklearn.tree import DecisionTreeClassifier


# In[34]:


from sklearn.metrics import accuracy_score


# Creating a pandas dataframe for the Boolean example provided in class

# In[2]:


T1 = pd.DataFrame({"X1" : [0,0,1,1,1,1,0,0],
"X2" : [0,1,0,1,0,1,1,0],
"X3" : [0,0,1,0,0,1,1,1],
"Y" : [1,1,0,1,1,0,0,0]})


# Display the entire DataFrame

# In[3]:


T1


# Separating X and Y DataFrame 

# In[5]:


X = T1.drop("Y", axis=1)


# In[6]:


X


# In[11]:


T1.Y


# In[14]:


Y = pd.DataFrame(T1.Y)


# In[15]:


Y


# Display only the attribute or column names of the dataframes

# In[26]:


list(T1.columns.values)


# In[28]:


list(X.columns.values)


# In[29]:


list(Y.columns.values)


# Spliting the data into training and test sets

# In[21]:


from sklearn.model_selection import train_test_split


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)


# Display X_train and y_train

# In[23]:


X_train


# In[25]:


y_train


# Using Decision Tree

# In[30]:


dtree = DecisionTreeClassifier()


# Build a model for the training examples

# In[32]:


dtree.fit(X_train, y_train)


# Predicting the values for Y

# In[33]:


dtree.predict(X_test)


# In[35]:


test=pd.DataFrame(dtree.predict(X_test))


# Calculating accuracy

# In[36]:


accuracy_score(test, y_test)


# In[37]:


y_test


# In[38]:


test

