#!/usr/bin/env python
# coding: utf-8

# ### Importing the libraries

# In[1]:


import numpy as np # linear algebra
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.mlab as mlab
import seaborn as sns 
from scipy.optimize import curve_fit
from IPython.display import display, HTML

get_ipython().run_line_magic('matplotlib', 'inline')


# ### Loading the dataset

# In[2]:


order_products_train = pd.read_csv("order_products__train.csv")
order_products_prior = pd.read_csv("order_products__prior.csv")
orders = pd.read_csv("orders.csv")
products = pd.read_csv("products.csv")
aisles = pd.read_csv("aisles.csv")
departments = pd.read_csv("departments.csv")
sample_submission = pd.read_csv("sample_submission.csv")


# ### Check the size of the 'orders' and 'products' files

# In[3]:


print(orders.shape)
print(products.shape) 


# ### There are around 3.5 million orders and around 50000 products

# ### We check the order information

# In[4]:


orders.info()


# In[5]:


orders.head()


# ### Number of orders per customer from max to min

# In[5]:


sns.set_style('dark')
customer_no = orders.groupby("user_id", as_index = False)["order_number"].max() 
customer_no


# In[6]:


num_bins = 10
n, bins, patches = plt.hist(customer_no["order_number"] , num_bins, color='blue', alpha=0.5)

plt.xlabel("No. of Orders")
plt.ylabel("Count")
plt.title("Number of Orders per Customer")


# ### Check the statistics of the customer data

# In[7]:


## MEAN

mean_value = customer_no["order_number"].mean()
mean_value


# In[8]:


## Standard Deviation

std_deviation = customer_no["order_number"].std()
std_deviation


# ### What are the most frequently purchased products in the dataset ?
# ### OR
# ### What products do consumers buy most often ?

# In[9]:


## Merging the train and prior datasets

t_p = order_products_train.append(order_products_prior,ignore_index = True)
prod = t_p.groupby("product_id",as_index = False)["order_id"].count() 


# ### Check the total number of products present in the dataset (Train + Prior)

# In[10]:


prod


# In[11]:


top = 20
product_Count = prod.sort_values("order_id",ascending = False)
df1 = product_Count.iloc[0:top,:]
df1 = df1.merge(products,on = "product_id")
display(df1.loc[:,["product_name"]])


# ### By finding the most frequently purchased products we make the following observations 
# 
# #### 1) Produce has the most demand and consumers tend to buy produce more often. This could be because produce consists of perishable items.
# #### 2) People buy organic food, despite the fact that it is expensive

# ### How many products does each department have ?

# In[12]:


x = pd.merge(left=products, right=departments, how='left')
lists = pd.merge(left = x, right=aisles, how='left')
lists 


# In[13]:


# Count the total number of products present in each department
group_list = lists.groupby("department")["product_id"].aggregate({'Total_products': 'count'}) 
group_list


# ### We try to explore the total number of products present in each department in a descending order

# In[29]:


final = group_list.reset_index() 
final.sort_values(by='Total_products', ascending=False, inplace=True)
final


# In[31]:


sns.set_style('white') 
ax = sns.barplot(x="Total_products", y="department", data=final,color = 'gray' )
#fig, ax = plt.subplots()

r = ax.spines["right"].set_visible(False)
#l = ax.spines["left"].set_visible(False)
t = ax.spines["top"].set_visible(False)


# ### We make the following observations -
# 
# #### 1) Personal care has the maximum number of products. This could be because personal care has many sub-categories like health-care items, cosmetics, deodrants, skin care products, bathroom essentials, etc. All these sub-categories consist of a huge variety of products and hence the count is maximum.
# 
# #### 2) Bulk department has the least number of products. This could be because it has very few items as consumers prefer to buy such items from a store.

# In[16]:


my_range=list(range(1,len(final.index)+1))


# ### We also plot and observe a bubble plot for the products in each department

# In[17]:


fig, ax = plt.subplots(figsize=(5,3.5))
plt.hlines(y=my_range, xmin=0, xmax=final['Total_products'], color='#007acc', alpha=0.2, linewidth=5)
plt.plot(final['Total_products'], my_range, "o", markersize=5, color='#007acc', alpha=0.6)


# ### Check the order_products_prior and order_products_train dataset

# In[18]:


order_products_prior.head()


# In[19]:


order_products_train.head()


# In[ ]:





# In[21]:


order_products_prior = pd.merge(order_products_prior, products, on='product_id', how='left')
order_products_prior.head() 


# In[22]:


order_products_prior = pd.merge(order_products_prior, aisles, on='aisle_id', how='left')
order_products_prior.head() 


# In[24]:


order_products_prior = pd.merge(order_products_prior, departments, on='department_id', how='left')
order_products_prior.head() 


# ### Create a new dataframe consisting of 'add_to_cart_order' and 'reordered' products from the prior set

# In[25]:


new_df = pd.DataFrame({'Add_to_cart': order_products_prior.add_to_cart_order, 'Reordered':order_products_prior.reordered })
new_df


# ### Which department has the highest and lowest number of reordered items ?
# 

# In[33]:


## Group the departments by the reordered items . Take an average of reordered per department to find the department 
## with the maximum number of reordered items

df2 = order_products_prior.groupby(["department"])["reordered"].aggregate("mean").reset_index()

plt.figure(figsize=(12,8))
sns.set_style('white')

ax1 = sns.scatterplot(df2['reordered'].values,df2['department'].values , color = 'gray')
plt.ylabel('Department', fontsize=15)
plt.xlabel('Reorder Ratio' , fontsize=15)
plt.title("Department wise reorder ratio", fontsize=15)
plt.xticks(rotation='horizontal')
r = ax1.spines["right"].set_visible(False)
t = ax1.spines["top"].set_visible(False)
plt.show() 


# ### We conclude that
# 
# #### 1) Personal care department has the lowest number of reordered items or the lowest reordered ratio.
# #### 2) Dairy eggs have department have the highest number of reordered items or the highest reordered ratio.
