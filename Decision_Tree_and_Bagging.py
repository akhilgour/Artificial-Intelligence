
# coding: utf-8

# In[1]:


from sklearn import datasets

wine = datasets.load_wine()


# In[9]:


wine.feature_names


# In[11]:


wine.target_names


# In[12]:


from sklearn.datasets import load_wine
X, Y = load_wine(return_X_y=True)


# Dimensions of X and Y

# In[14]:


len(X)


# In[17]:


len(Y)


# In[31]:


X.shape


# In[32]:


Y.shape


# In[33]:


X.ndim


# In[34]:


Y.ndim


# Type of X & Y

# In[13]:


type(X)


# In[16]:


type(Y)


# Decision tree classifier

# In[19]:


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, Y,
test_size=0.30, random_state=1)


# random_state is a parameter which is used to get consistent results. If we pass 1 to it, then every time we run the code, we get the same output. If it is set to NONE, each time random seed will be chosen, and we'll get different results. (Source: stackoverflow.com)

# In[20]:


from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()


# In[21]:


dtree.fit(X_train, y_train)


# In[22]:


y_pred_test = dtree.predict(X_test)


# In[24]:


y_pred_train = dtree.predict(X_train)


# In[25]:


from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' %accuracy_score(y_test, y_pred_test))


# In[30]:


print('Accuracy: %.2f' %accuracy_score(y_train, y_pred_train))


# Training data accuracy is better.

# Bagging

# In[35]:


from sklearn.ensemble import BaggingClassifier

tree = DecisionTreeClassifier(criterion='gini', random_state=100, max_depth=None)
bag = BaggingClassifier(base_estimator=tree, n_estimators=100, max_samples=1.0, max_features=4, bootstrap=True,
bootstrap_features=False, n_jobs=1, random_state=1)


# In[47]:


bag.fit(X_train, y_train)


# In[48]:


y_pred_test = bag.predict(X_test)


# In[49]:


y_pred_train = bag.predict(X_train)


# In[50]:


from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' %accuracy_score(y_test, y_pred_test))


# In[51]:


print('Accuracy: %.2f' %accuracy_score(y_train, y_pred_train))


# Yes, the bagged classifier improved on the accuracy
