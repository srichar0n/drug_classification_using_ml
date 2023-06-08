#!/usr/bin/env python
# coding: utf-8

# In[1]:


#drug classification


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv("C:\\Users\\Sricharan Reddy\\Downloads\\drug200.csv")


# In[4]:


#data set had been loaded 


# In[5]:


df.head()


# In[6]:


df.describe()


# In[7]:


df.shape


# In[8]:


df['Sex'].value_counts()


# In[9]:


df['Cholesterol'].unique()


# In[10]:


df.isnull().sum()


# In[11]:


#there no null values


# In[12]:


df.duplicated().sum()


# In[13]:


#there are no duplicates 


# In[14]:


df.info()


# In[15]:


#df = pd.get_dummies(df,drop_first=True)


# In[16]:


y=df['Drug']


# In[17]:


#df = pd.get_dummies(df,drop_first=True)


# In[18]:


df.drop(columns=['Drug'],inplace=True)


# In[19]:


x = pd.get_dummies(df,drop_first=True)


# In[20]:


df.head()


# In[21]:


from sklearn.model_selection import train_test_split


# In[22]:


x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=9)


# In[23]:


from sklearn.tree import DecisionTreeClassifier


# In[24]:


model = DecisionTreeClassifier()


# In[25]:


model


# In[26]:


model.fit(x_train,y_train)


# In[27]:


y_pred_train = model.predict(x_train)


# In[28]:


y_pred_test = model.predict(x_test)


# In[29]:


from sklearn.metrics import accuracy_score


# In[30]:


print(accuracy_score(y_train,y_pred_train))


# In[31]:


print(accuracy_score(y_test,y_pred_test))


# In[32]:


from sklearn.model_selection import cross_val_score


# In[33]:


print(cross_val_score(model,x,y,cv=5).mean())


# In[34]:


y.unique()


# In[35]:


#cross validation score is 98.5%


# In[36]:


from sklearn.tree import plot_tree
plt.figure(figsize=(10,8),dpi=300)

plot_tree(model,filled=True,feature_names=x.columns,class_names=['DrugY', 'drugC', 'drugX', 'drugA', 'drugB'])

plt.show()


# In[37]:


from sklearn.model_selection import GridSearchCV


# In[38]:


estimator = DecisionTreeClassifier(random_state=9)


# In[39]:


param = {"criterion":['gini','entropy'],'max_depth':[1,2,3,4,5]}


# In[40]:


grid = GridSearchCV(estimator , param ,scoring="accuracy",cv=5)


# In[41]:


grid.fit(x_train,y_train)


# In[42]:


grid.best_estimator_


# In[43]:


grid.best_estimator_.feature_importances_


# In[44]:


x.head()


# In[47]:


type(x)


# In[48]:


x.drop(columns=['Sex_M'],inplace=True)


# In[49]:


x.head()


# In[50]:


model = DecisionTreeClassifier(max_depth=4)


# In[51]:


model.fit(x_train,y_train)


# In[52]:


y_pred_train = model.predict(x_train)


# In[53]:


y_pred_test = model.predict(x_test)


# In[54]:


print(accuracy_score(y_pred_test,y_test))


# In[55]:


print(accuracy_score(y_pred_train,y_train))


# In[57]:


print(cross_val_score(model,x,y,cv=5).mean())


# In[58]:


#still the cross val score remains same after optimizing the algorithm


# In[59]:


#the end ---------------------------------------------------------


# In[60]:


#thank youuuuu


# In[ ]:




