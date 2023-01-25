#!/usr/bin/env python
# coding: utf-8

# # Predict the percentage of an student based on the no. of study hours.

# Step 1: Importing Python llibrary 

# In[1]:


import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# Step 2 : Importing data-set  

# In[2]:


data = pd.read_csv("data.csv")
print("Importing Data Successfully")


# Step 3 : check whether Data imported successfully or not ?

# In[3]:


print("For this we print first 10 data of Data-set ")
data.head(10)


# In[4]:


print("You have correctly imported Data-set")


# # step 4 : Plot the Graph 📊 ,for detail Analysis of Data-set 

# In[5]:


data.plot(x='Hours',y='Scores',style='1')
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# In[6]:


data.plot.pie(x='Hours',y='Scores')


# In[7]:


data.plot.scatter(x='Hours',y='Scores')


# In[8]:


data.plot.bar(x='Hours',y='Scores')


# In[9]:


# data.sort_values(["Hours"], axis=0,
#                  ascending=[True],inplace=True)
# data.head(10)
# data.plot.bar(x='Hours',y='Scores')


# # After ploting different graph , we have observed that as Study Hours increases , score is also increase. Which is good sign of correct data. In our daily life , we have observe the same phenomenon. 

# Step 5 : Now, we have prepared tha data for our model

# In[10]:


X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values  
# print(X)


# Step 6 : Now , we have divide the data for tarining & testing the model 

# In[11]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


# # Training the Algorithm

# In[12]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression() 


# from sklearn.ensemble import RandomForestRegressor
# regressor = RandomForestRegressor(n_estimators = 1000, random_state = 42)


regressor.fit(X_train, y_train) 
print("Training complete.")


# In[13]:


# Plotting the regression line
# line = regressor.coef_*X+regressor.intercept_

# # Plotting for the test data
# plt.scatter(X, y)
# plt.plot(X, line);
# plt.show()


# Step 7 : Now, our Model is ready . Its time to test it .

# In[14]:


print(X_test) 
print("Predection of Score")
y_pred = regressor.predict(X_test)
print(y_pred)


# Step 8 : Now, Checking the accuracy of our model

# In[15]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  

df 


# Step 9 : Now, Its times to prediction with custom input

# In[16]:


hours = [[9.25]]
pred = regressor.predict(hours)
print(pred)


# # Evaluating the model

# In[17]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 


# In[ ]:




