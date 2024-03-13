#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# ## DataTypes & Attributes

# In[4]:


# Numpy's main datatype is ndarray
a1 = np.array([1,2,3])
a1


# In[5]:


type(a1)


# In[57]:


a2 = np.array([[1,2,3.3],[2,5,6]])

a3 = np.array([[[1,2,3],
              [4,5,6],
              [7,8,9]],
             [[10,11,12],
             [13,14,15],
             [16,17,18]]])


# In[7]:


a2


# In[8]:


a3


# In[9]:


a1.shape


# In[10]:


a2.shape


# In[11]:


a3.shape


# In[12]:


a1.ndim 


# In[13]:


a2.ndim , a3.ndim


# In[14]:


a3.dtype


# In[15]:


a1.size , a2.size , a3.size


# In[16]:


type(a1)


# In[22]:


# Create a DataFrame from  a Numpy array
import pandas as pd

df = pd.DataFrame(a2)
df


# ## 2.Creating Arrays

# In[23]:


sample_array = np.array([1,2,3])
sample_array


# In[24]:


sample_array.dtype


# In[27]:


ones = np.ones((2,3))
ones 


# In[28]:


ones.dtype


# In[29]:


zeros = np.zeros((2,4))


# In[30]:


zeros


# In[31]:


range_array = np.arange(0,10,2)
range_array


# In[33]:


random_array = np.random.randint(0,10,size = (3,5))
random_array


# In[34]:


random_array.size


# In[35]:


random_array.shape


# In[41]:


random_array2 = np.random.random((5,3))
random_array2


# In[42]:


# Pseudo- Random numbers

## np.random.seed()

random_array3 = np.random.randint(10 , size = (5,3))
random_array3


# In[43]:


random_array3.shape


# In[44]:


np.random.seed(7)
random_array4 = np.random.random((5,3))
random_array4


# # Viewing arrays and matrices

# In[47]:


random_array3


# In[46]:


np.unique(random_array3)


# In[48]:


a4 = np.random.randint(10,size = (2,3,4,5))
a4


# In[49]:


# get the first 4 number of the innermost arrays
a4[:,:,:,:4]


# # 4.Manipulating  and comparing arrays

# #Arithmatic

# In[50]:


a1


# In[51]:


ones = np.ones(3)
ones


# In[52]:


a1 + ones


# In[53]:


a1 - ones 


# In[54]:


a1 * ones 


# In[55]:


a2


# In[58]:


a1*a2


# In[59]:


a2/a1


# In[60]:


# floor division removes the decimals (rounds down)
a2//a1


# In[61]:


a2 


# In[62]:


a2**2


# In[63]:


np.square(a2)


# In[64]:


np.add(a1,a2)


# In[65]:


a1 % 2 


# ## Aggregation 
# 
# Aggregation = performing the same operation on a number of things

# In[67]:


listy_list = [1,2,3]
type(listy_list)


# In[68]:


sum(listy_list)


# In[69]:


sum(a1)


# In[70]:


np.sum(a1)


# In[72]:


# use Python's methods ('sum()') on python datatypes and use Numpy's methods on 
# Numpy arrays (np.sum()).


# In[73]:


# Create a massive Numpy array
massive_array = np.random.random(10000)
massive_array.size


# In[75]:


massive_array[:100]


# In[81]:


get_ipython().run_line_magic('timeit', ' sum(massive_array)  # pythons sum')
get_ipython().run_line_magic('timeit', " np.sum(massive_array) # Numpy's sum")


# In[82]:


a2


# In[83]:


np.mean(a2)


# In[84]:


np.min(a2)


# In[85]:


np.std(a2)


# In[86]:


# Variance = measure of the average degree to which each number is different 
# to the mean 
# Higher variance = wider range of number 
np.var(a2)


# In[87]:


# Standard deviation = squareroot of variance 
np.sqrt(np.var(a2))


# In[89]:


# Demo of std and var
high_var_array = np.array([1,100,200,300,4000,5000])
low_var_array = np.array([2,4,6,8,10])


# In[90]:


np.var(high_var_array) , np.var(low_var_array)


# In[91]:


np.std(high_var_array) , np.std(low_var_array)


# In[92]:


np.mean(high_var_array) , np.mean(low_var_array)


# In[95]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.hist(high_var_array)
plt.show()


# In[96]:


plt.hist(low_var_array)
plt.show()


# ### Reshaping and Transposing 

# In[97]:


a2


# In[98]:


a2.shape


# In[99]:


a3


# In[101]:


a2.shape


# In[102]:


a2.reshape(2,3,1).shape


# In[103]:


a3.shape


# In[104]:


a2_reshape = a2.reshape(2,3,1)
a2_reshape


# In[105]:


a2_reshape * a3


# In[107]:


a2


# In[106]:


# Transpose switches the axis 
a2.T


# In[108]:


a2.T.shape


# ## Dot Product 

# In[109]:


np.random.seed(0)

mat1 = np.random.randint(10,size =(5,3))
mat2 = np.random.randint(10,size = (5,3))

mat1


# In[110]:


mat2


# In[111]:


mat1.shape , mat2.shape


# In[112]:


mat1*mat2


# In[114]:


mat2.T


# In[116]:


# Dot product 
mat3 = np.dot(mat1,mat2.T)
mat3


# In[117]:


mat3.shape


# ## Dot prduct example

# In[118]:


np.random.seed(0)
# Number of jars sold 
sales_amounts = np.random.randint(20,size=(5,3))
sales_amounts


# In[120]:


# create weekly sales Dataframe
weekly_sales = pd.DataFrame(sales_amounts,
                           index = ["mon","tues","wed","thurs","fri"],
                           columns = ["Almond butter","peanut butter","cashew butter"])
weekly_sales


# In[153]:


# Create prices array
prices = np.array([10, 8, 12])
prices 


# In[127]:


# prices DataFrame
butter_prices = pd.DataFrame(prices.reshape(1,3),
            index = ["price"],
            columns = ["Almond butter","Peanut butter","cashew butter"])
butter_prices


# In[154]:


total_sales = prices.dot(sales_amounts)


# In[125]:


# shapes are'nt aligned , lets transpose
total_sales = prices.dot(sales_amounts.T)
total_sales 


# In[128]:


# Create daily_sales
butter_prices


# In[129]:


sales_amounts.shape


# In[132]:


total_sales = prices.dot(sales_amounts.T)
total_sales


# In[147]:


# Create daily_sales |
butter_prices.shape , weekly_sales.shape


# In[142]:


weekly_sales


# In[151]:


butter_prices.shape , weekly_sales.shape


# In[159]:


weekly_sales2  = weekly_sales.T
weekly_sales2 , weekly_sales2.shape


# In[165]:


weekly_sales.shape , butter_prices.T.shape


# In[166]:


daily_sale = weekly_sales.dot(butter_prices.T)
daily_sale


# ### Comparision operators

# In[167]:


a1


# In[168]:


a2


# In[169]:


a1>a2


# In[170]:


a1>=a2


# In[171]:


a1 <4


# In[172]:


a1 == a2


# ## 5. Sorting arrays 

# In[177]:


random_array = np.random.randint(10,size =(3,5) )
random_array


# In[179]:


np.sort(random_array)


# In[181]:


np.argsort(random_array)


# In[185]:


np.argmax(random_array,axis = 1)


# ## 6. Practical Example- Numpy IN Action !!!!

# In[197]:


import pandas as pd


# <img src="images/panda.png"/>

# In[200]:


# Turn an image into a Numpy array
from matplotlib.image import imread

panda = imread("images/panda.png")
print(type(panda))


# In[201]:


panda.size , panda.shape , panda.ndim


# <img src = "images/car-photo.png"/>

# In[202]:


car = imread("images/car-photo.png")
print(type(car))


# <img src="images/dog-photo.png"/>

# In[203]:


dog = imread("images/dog-photo.png")
print(type(dog))


# In[204]:


dog


# In[ ]:




