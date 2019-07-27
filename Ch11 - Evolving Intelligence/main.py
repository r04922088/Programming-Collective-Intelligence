#!/usr/bin/env python
# coding: utf-8

# In[13]:


import geneticProgramming
from random import random, randint, choice


# In[7]:


# make training data x and ground truth y


# In[8]:


def hiddenfunction(x, y):
    """ a function that able to generate a set of training data depend on xy and return ground truth
        the input paramator's count of this function is depend on input length(pc)
    """
    return x**2 + 2*y + 3*x + 5


# In[9]:


def buildhiddenset():
    """ a function that build a list of training dataset and groundtruth
    
    """
    rows = []
    for i in range(200):
        x = randint(0, 40)
        y = randint(0, 40)
        rows.append([x, y, hiddenfunction(x, y)])
    return rows


# In[14]:


rf = geneticProgramming.getrankfunction(buildhiddenset())


# In[ ]:


geneticProgramming.evolve(2, 500, rf, mutationrate=0.2, breedingrate=0.1, pexp=0.7, pnew=0.1)


# In[ ]:




