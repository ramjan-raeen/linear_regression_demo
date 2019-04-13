#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[2]:


dataframe = pd.read_csv('challenge_dataset.txt')
x_values = dataframe[['x1']]
y_values = dataframe[['x2']]


# In[3]:


linearReg = linear_model.LinearRegression()
linearReg.fit(x_values, y_values)


# In[4]:


plt.scatter(x_values, y_values)
plt.plot(x_values, linearReg.predict(x_values))
plt.show()


# In[ ]:




