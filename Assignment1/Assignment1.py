#!/usr/bin/env python
# coding: utf-8

# In[69]:


import numpy as np
import matplotlib.pyplot as plt


# In[70]:


data = np.loadtxt('data_group6.txt')


# In[71]:


data.shape


# In[72]:


plt.scatter(data[:,3], data[:,2])
plt.title('Sunspot number to flux ration dependency')
plt.xlabel('monthly mean sunspot number')
plt.ylabel('solar ratio flux F10.7cm')
plt.legend()
plt.show()


# In[135]:


sunspot_num_smth = np.empty(data.shape[0])
solar_flux_smth = np.empty(data.shape[0])
sunspot_num_smth[:6] = np.ones([1, 6]) * np.mean(data[:6,3])
sunspot_num_smth[-6:] = np.ones([1, 6]) * np.mean(data[-6:,3])
solar_flux_smth[:6] = np.ones([1, 6]) * np.mean(data[:6,2])
solar_flux_smth[-6:] = np.ones([1, 6]) * np.mean(data[-6:,2])
for i in range(6, data.shape[0] - 6):
    sunspot_num_smth[i] = 1/24 * (data[i-6][3] + data[i+6][3]) + 1/12 * (np.sum(data[i-5:i+6, 3]))
    solar_flux_smth[i] = 1/24 * (data[i-6][2] + data[i+6][2]) + 1/12 * (np.sum(data[i-5:i+6, 2]))


# In[136]:


sunspot_num_smth


# In[137]:


plt.plot(data[:,0], sunspot_num_smth)
plt.plot(data[:,0], solar_flux_smth)
plt.show()


# In[152]:


F_exp = data[:, 2]
F = np.empty(F_exp.shape)
R = np.ones([data.shape[0], 4])
beta = np.ones(4)
for i in range(4):
    R[:, i] = data[:, 3]**i
beta = np.matmul(np.linalg.inv(np.matmul(R.T, R)), R.T).dot(F_exp)


# In[148]:


beta


# In[159]:


F = np.ones(data.shape[0])*beta[0]+beta[1]*data[:,3]+beta[2]*data[:,3]**2+beta[3]*data[:,3]**3


# In[160]:


plt.plot(data[:,0], sunspot_num_smth)
plt.plot(data[:,0], F)
plt.show()


# In[ ]:




