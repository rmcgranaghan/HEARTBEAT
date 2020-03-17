#!/usr/bin/env python
# coding: utf-8

# # Variable Importance Python Jupyter Notebook. 

# This juputer notebook provides an  example to estimate variable importance using 
# the Random Forest machine learing method. The code doesn't run by itself,  it just shows how to impliment the random forest for variable selection.  A separate python script of the same code is also provided.  

# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats import norm
from scipy import stats
# import warnings
# warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


#Train and test split of the trainig and target. You might do your own split,  
#example first three years as trinig and last year test.
x_train, x_test, y_train, y_test = train_test_split(training, target)


# In[ ]:


#Random Forest Regressor
# Modify this for your experiment
modelRF=RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=None, min_samples_split=2) 


# In[ ]:


#Fit Model 
modelRF=modelRF.fit(x_train,y_train)


# In[ ]:


#Predict on the test
y_predict=modelRF.predict(x_test)


# In[ ]:


#Variable Importance estimation and plotting. 
#Note:  In this case the x_train is a pandas dataframe with the names of columns the same as the feature names.
var_importance=modelRF.feature_importances_
important_idx = np.where(var_importance > 0)
important_var = x_train.columns[important_idx]
sorted_idx = np.argsort(var_importance[important_idx])[::1]
var_importance=var_importance[important_idx]
sorted_var=important_var[sorted_idx]
sorted_var_importance=var_importance[sorted_idx]
pos = np.arange(sorted_idx.shape[0]) + .5
plt.figure(figsize=(18, 15))
plt.barh(pos, sorted_var_importance, color='red',align='center')
plt.yticks(pos, sorted_var,fontsize=20)
plt.xlabel('Relative Importance',fontsize=30)
plt.title('Variable Importance',fontsize=30)
plt.rcParams['ytick.labelsize']=30
plt.rcParams['xtick.labelsize']=30
plt.tight_layout()
plt.margins(y=0)
plt.savefig('VariableImportance.png', facecolor='w', format='png')
plt.show()


# ![The Folowing](VariableImportance.png "Feautures Used")

# In[14]:


pwd


# In[15]:


pwd


# In[ ]:




