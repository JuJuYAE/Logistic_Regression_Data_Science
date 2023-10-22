#!/usr/bin/env python
# coding: utf-8

# In[74]:


import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[75]:


df = pd.read_csv("titanic_train.csv")


# In[76]:


df.head()


# In[77]:


df.info


# In[78]:


sns.heatmap(df.isnull(), yticklabels = False, cbar = False, cmap = "viridis")


# In[79]:


sns.set_style("whitegrid")


# In[80]:


sns.countplot(x = "Survived", data = df)


# In[81]:


sns.countplot(x = "Survived", data = df, hue = "Sex", palette = "RdBu_r")


# In[82]:


sns.countplot(x = "Survived", data = df, hue = "Pclass")


# In[83]:


sns.displot(df["Age"].dropna(), bins = 30)


# In[84]:


df["Age"].plot.hist(bins = 35)


# In[85]:


df.info()


# In[86]:


sns.countplot(x = "SibSp", data = df)


# In[87]:


sns.displot(df["Fare"], bins = 35)


# In[88]:


import cufflinks as cf


# In[89]:


cf.go_offline()


# In[90]:


df["Fare"].iplot(kind = "hist", bins = 35)


# In[29]:


plt.figure(figsize = (10,7))
sns.boxplot(x = "Pclass", y = "Age", data = df)


# In[30]:


def impute_age(cols) : 
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age): 
        if Pclass == 1 : return 37 
        elif Pclass == 2 :return 29 
        else : return 24
    else : return Age 


# In[31]:


df["Age"] = df[["Age", "Pclass"]].apply(impute_age, axis = 1)


# In[32]:


sns.heatmap(df.isnull(), yticklabels = False, cbar = False, cmap = "viridis")


# In[33]:


df.drop("Cabin", axis = 1, inplace = True)


# In[34]:


df.head()


# In[35]:


sns.heatmap(df.isnull(), yticklabels = False, cbar = False, cmap = "viridis")


# In[36]:


df.dropna(inplace = True)


# In[41]:


sns.heatmap(df.isnull(), yticklabels = False, cbar = False, cmap = "viridis")


# In[44]:


sex = pd.get_dummies(df["Sex"],drop_first = True)


# In[45]:


sex


# In[46]:


embark = pd.get_dummies(df["Embarked"],drop_first = True)


# In[47]:


embark


# In[48]:


df = pd.concat([df,sex, embark], axis = 1)


# In[54]:


df.head()


# In[53]:


df.drop(["Sex", "Embarked", "Name", "Ticket"],axis = 1, inplace = True)


# In[ ]:


df.head()


# In[55]:


df.tail()


# In[56]:


df.drop(["PassengerId"],axis = 1, inplace = True)


# In[57]:


df.head()


# In[58]:


X = df.drop("Survived", axis = 1)
y = df["Survived"]


# In[59]:


from sklearn.model_selection import train_test_split


# In[60]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[62]:


from sklearn.linear_model import LogisticRegression


# In[63]:


logmodel = LogisticRegression()


# In[64]:


logmodel.fit(X_train, y_train)


# In[65]:


prediction = logmodel.predict(X_test)


# In[68]:


from sklearn.metrics import classification_report


# In[70]:


print(classification_report(y_test, prediction))


# In[71]:


from sklearn.metrics import confusion_matrix


# In[72]:


(confusion_matrix(y_test, prediction))


# In[ ]:




