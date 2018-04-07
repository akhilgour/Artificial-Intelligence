
# coding: utf-8

# In[1]:


from sklearn.datasets import make_blobs


# In[2]:


from sklearn.naive_bayes import GaussianNB


# In[3]:


X, y =make_blobs(n_samples=100, cluster_std=3.5, centers=2)


# In[4]:


import matplotlib.pyplot as plt


# In[5]:


plt.scatter(X[:, 0], X[:, 1],c=['blue','red'],)


# In[6]:


plt.xlabel('X axis')


# In[7]:


plt.ylabel('Y axis')


# In[8]:


plt.show()


# In[9]:


from sklearn.model_selection import train_test_split


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)


# In[11]:


from sklearn.naive_bayes import GaussianNB


# In[12]:


from sklearn.naive_bayes import GaussianNB



# In[13]:


clf = GaussianNB()


# In[14]:


clf.fit(X, y)
GaussianNB(priors=None)


# In[15]:


print(clf.predict(X_train))


# In[16]:


print(clf.predict(X_test))


# In[17]:


from sklearn.metrics import accuracy_score


# In[18]:


y_pred_test = clf.predict(X_test)


# In[19]:


print('Accuracy: %.2f' %accuracy_score(y_test, y_pred_test))

