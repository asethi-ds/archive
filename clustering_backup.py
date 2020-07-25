#!/usr/bin/env python
# coding: utf-8


# Sales Analytics
# Importing Packages
import time
start_time = time.time()
from kmodes.kprototypes import KPrototypes
from kmodes.kmodes import KModes
from kmodes.kmodes import KModes
import pandas as pd
import numpy as np
import datetime as dt
from configparser import SafeConfigParser, ConfigParser
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import gender_guesser.detector as gender#from gender_guesser import gender

#from genderator.detector import *

#passing config file information
config_file_name='file_params.ini'

parser = ConfigParser()
parser.read(config_file_name)

#Getting source directory path and importing files
source_dir_path= parser.get('paths', 'source_dir')
source_dir = os.listdir(source_dir_path )

if(len(source_dir)):

    all_files = glob.glob(source_dir_path + "/*.csv")
    orders_data=pd.read_csv(all_files[1])
    orders_line_items_data=pd.read_csv(all_files[2])
    customers_data=pd.read_csv(all_files[0])
    products_data=pd.read_csv(all_files[3])
    customers_age=pd.read_csv('customer_id_age.csv')
    product_map_division=pd.read_excel('division_map.xlsx')
    

#Load the required packages for exploratory analysis and data plotting
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10, 5)
plt.style.use('ggplot')


#Getting data for customers age
customers_data=pd.merge(customers_data, customers_age, on='customer_id', how='left')

#Assigning gender
d = gender.Detector()
def assign_gender_first_name(first_name):
    return d.get_gender(first_name)

customers_data['gender']=customers_data['first_name'].apply(assign_gender_first_name)

#Extracting relevant columns from customers dataset
customer_workset=customers_data[['customer_id','gender','age','tags','total_spent','order_count','default_address_country','default_address_zip','default_address_province','default_address_city']]

# Plotting histogram of cumulative distribution of total spend of the customers
mu = 200
sigma = 25
n_bins = 30

# 
plot_series_total_spend=customer_workset['total_spent']
plot_series_order_count=customer_workset['order_count']


fig, ax = plt.subplots(figsize=(8, 4))

# plot the cumulative histogram
n, bins, patches = ax.hist(plot_series_order_count, n_bins, density=True, histtype='step',
                           cumulative=True, label='Empirical')

# tidy up the figure
ax.grid(True)
ax.legend(loc='right')
ax.set_title('Cumulative step histograms')
ax.set_xlabel('Total Spent Amount')
ax.set_ylabel('Likelihood of occurrence')

# Plotting the histogram of cumulative distribution of Orders Count of all the customer
plt.hist(plot_series_order_count, density=True, bins=80)
plt.ylabel('Probability of occurance');
plt.xlabel('Total Order Count');

# Dividing the total orders into buckets
bins = np.array([0,1,5,10,120])
customer_workset["total_orders_bucket"] = pd.cut(customer_workset.order_count, bins)


# Checking the frequency distribution
customer_workset.total_orders_bucket.value_counts()

#Dividing the total spend into different columns
bins2 = np.array([0,500,3000,15000])
customer_workset["total_spent_bucket"] = pd.cut(customer_workset.total_spent, bins2)
customer_workset.total_spent_bucket.value_counts()

# Calculating the average customer expenditure per order
customer_workset['average_customer_spent'] = customer_workset.apply(lambda row: row.total_spent / 
                                  (row.order_count), axis = 1) 

plot_series_customer_spent=customer_workset['average_customer_spent']

# Plottiing histogram of the cumulative probability distribution of the average spend per customer
plt.hist(plot_series_customer_spent, density=True, bins=300)
plt.ylabel('Probablity of occurance');
plt.xlabel('Average Spent per customer');

bins2 = np.array([0,40,100,400,2000])
customer_workset["average_customer_spent_bucket"] = pd.cut(customer_workset.average_customer_spent, bins2)
customer_workset.average_customer_spent_bucket.value_counts()

#Getting division by joining with skew id
orders_line_items_data=pd.merge(orders_line_items_data, product_map_division, on='sku', how='left')

# Getting relevant variables from order line items, orders and product table
orders_data_relevant=orders_data[['order_id','financial_status','customer_id','app_id','landing_site','buyer_accepts_marketing','buyer_accepts_marketing']]
orders_line_items_data_relevant=orders_line_items_data[['order_id','line_item_id','product_id','gift_card','product_gender']]
product_data_relevant=products_data[['product_id','product_type']]

#Merging the tables to order line level information for all customers
orders_merged=pd.merge(orders_data_relevant, orders_line_items_data_relevant, on='order_id', how='left')
product_orders_merged= pd.merge(orders_merged, product_data_relevant, on='product_id', how='left')
workset_main=pd.merge(customer_workset, product_orders_merged, on='customer_id', how='left')

#Get unique customer ids
df2 = pd.DataFrame({'customer_id':workset_main.customer_id.unique()})

# Get all product categories ordered by each customer
df2['product_type_all'] = [list(set(workset_main['product_type'].loc[workset_main['customer_id'] == x['customer_id']])) 
    for _, x in df2.iterrows()]

#adding gender to customer id 
all_prod=pd.merge(df2, workset_main[['gender','customer_id']], on='customer_id', how='inner')
all_prod=all_prod.loc[all_prod.astype(str).drop_duplicates().index]
all_prod=all_prod.reset_index()

# identifying customers who buy only one product type
def only_product_type_classify(product_type_all):
   if(len(product_type_all)==1):
        return product_type_all
    
# identifying customers who buy only one product type
def only_gender_type_classify(product_gender_all):
   if(len(product_gender_all)==1):
        return product_gender_all
    
# Labelling the only product type
all_prod['only_product_type'] = all_prod['product_type_all'].apply(only_product_type_classify)
all_prod['only_product_type'] = all_prod['only_product_type'].str[0]


# In[375]:


#Getting the list of all product genders each customer purchase
df2['product_gender_all'] = [list(set(workset_main['product_gender'].loc[workset_main['customer_id'] == x['customer_id']])) 
for _, x in df2.iterrows()]


# In[376]:


#Getting gender
all_prod_gender=pd.merge(all_prod,df2[['customer_id','product_gender_all']], on='customer_id', how='inner')
all_prod_gender=all_prod_gender.loc[all_prod_gender.astype(str).drop_duplicates().index]
all_prod_gender=all_prod_gender.reset_index()


# In[377]:


all_prod_gender['only_gender_type'] = all_prod_gender['product_gender_all'].apply(only_gender_type_classify)


# In[378]:


all_prod_gender['only_gender_type'] = all_prod_gender['only_gender_type'].str[0]


# In[379]:


all_prod_gender['only_buys_one_gender']=np.where(all_prod_gender['only_gender_type'].isnull(),0,1)
all_prod_gender['only_buys_one_prd_type']=np.where(all_prod_gender['only_product_type'].isnull(),0,1)


# In[380]:


#strip whitespaces in the data
all_prod_gender = all_prod_gender.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Applying one hot encoding to engineered features
#COnverting category to binary flag
one_hot_gender=pd.get_dummies(all_prod_gender['only_gender_type'], prefix='only_gender_type')
# Applying the o-h-e to gender
all_prod_gender=all_prod_gender.join(one_hot_gender)

# Applying the o-h-e to product type
one_hot_product_type=pd.get_dummies(all_prod_gender['only_product_type'], prefix='only_product_type')
# Applying the o-h-e to product type
all_prod_gender=all_prod_gender.join(one_hot_product_type)


# Replacing mostly male with male
# Replacing mostly female with female
# Replacing unknown with neutral
all_prod_gender.loc[all_prod_gender['gender'] == 'mostly_male', 'gender'] = 'male'
all_prod_gender.loc[all_prod_gender['gender'] == 'mostly_female', 'gender'] = 'female'
all_prod_gender.loc[all_prod_gender['gender'] == 'unknown', 'gender'] = 'neutral'


# In[397]:


#all_prod_gender


# In[385]:


#Removing duplicate columns 
workset_main = workset_main.loc[:,~workset_main.columns.duplicated()]

#joinig with the previously created columns
sample_out=pd.merge(all_prod_gender,workset_main[['age','default_address_province',
       'total_orders_bucket', 'total_spent_bucket', 'average_customer_spent_bucket','buyer_accepts_marketing','customer_id']],on='customer_id',how='left' )

sample_out=sample_out.drop_duplicates()

bins = np.array([0,25,50,120])
sample_out["age_bucket"] = pd.cut(sample_out.age, bins)
#


# In[388]:


# Labelling the latest flag for accepts marketting
sample_out['accepts_marketing'] = sample_out.groupby('customer_id')['buyer_accepts_marketing'].transform('max')
sample_out=sample_out.drop(['buyer_accepts_marketing'], axis=1)


# In[396]:


sample_out=sample_out.drop_duplicates()


# In[417]:


states_map=pd.read_excel('states_map.xlsx')
states_map.columns


# In[418]:


sample_out=sample_out.rename(columns={"default_address_province": "State"})


# In[419]:


customer_sales_data_out=pd.merge(sample_out, states_map[['State','Region']], on='State', how='left')


# In[421]:


customer_sales_data_out.columns


# In[ ]:


customer_sales_data_out.to_excel('customer_sales_data_out.xlsx')


# In[251]:





# In[253]:





# In[ ]:





# In[292]:





# In[ ]:





# In[294]:





# In[295]:





# In[296]:





# In[297]:





# In[331]:


#all_prod_gender


# In[ ]:


#all_prod_gender=all_prod_gender[['customer_id','gender','only_product_type','only_buys_gender']]


# In[327]:





# In[332]:





# In[ ]:





# In[ ]:





# In[337]:





# In[ ]:





# In[ ]:





# In[ ]:


variable_set_zero=['total_orders_bucket','total_spent_bucket','average_customer_spent_bucket','age_bucket','buyer_accepts_marketing']
variable_set_one=['total_orders_bucket','total_spent_bucket','average_customer_spent_bucket','buyer_accepts_marketing','age_bucket','gender']
variable_set_two=['only_buys_gender_Children','only_buys_gender_Men','only_buys_gender_Women','only_product_type_Athleisure','only_product_type_Dress Shirt','only_product_type_Final Sale','only_product_type_Gift Card','only_product_type_Long-johns','only_product_type_Loungewear','only_product_type_PrePack','only_product_type_Socks','only_product_type_T-shirts','only_product_type_Un-Tucked Shirt','only_product_type_Undershirts','only_product_type_Underwear','default_address_province','total_orders_bucket','total_spent_bucket','average_customer_spent_bucket','buyer_accepts_marketing','age_bucket','gender']


# In[ ]:


sample_sales_features[['total_orders_bucket']] = sample_sales_features[['total_orders_bucket']].fillna(sample_sales_features[['total_orders_bucket']].mode().iloc[0])
sample_sales_features[['total_spent_bucket']] = sample_sales_features[['total_spent_bucket']].fillna(sample_sales_features[['total_spent_bucket']].mode().iloc[0])
sample_sales_features[['average_customer_spent_bucket']] = sample_sales_features[['average_customer_spent_bucket']].fillna(sample_sales_features[['average_customer_spent_bucket']].mode().iloc[0])
sample_sales_features[['age_bucket']] = sample_sales_features[['age_bucket']].fillna(sample_sales_features[['age_bucket']].mode().iloc[0])


kproto = KPrototypes(n_clusters=6, init='Cao', verbose=2)
clusters = kproto.fit_predict(sample_sales_features[variable_set_one], categorical=[0,1, 2,4,5])
sample_sales_features['clust_1']=clusters
sample_sales_features.to_excel('sample_sales_features_cluster_output.xlsx')


# In[ ]:


# variable_set_one=['total_orders_bucket','total_spent_bucket','average_customer_spent_bucket','buyer_accepts_marketing','age_bucket','gender']
# variable_set_two=['only_buys_gender_Children','only_buys_gender_Men','only_buys_gender_Women','only_product_type_Athleisure','only_product_type_Dress Shirt','only_product_type_Final Sale','only_product_type_Gift Card','only_product_type_Long-johns','only_product_type_Loungewear','only_product_type_PrePack','only_product_type_Socks','only_product_type_T-shirts','only_product_type_Un-Tucked Shirt','only_product_type_Undershirts','only_product_type_Underwear','default_address_province','total_orders_bucket','total_spent_bucket','average_customer_spent_bucket','buyer_accepts_marketing','age_bucket','gender']
# variable_set_three=['total_orders_bucket','total_spent_bucket','average_customer_spent_bucket','buyer_accepts_marketing','age_bucket','gender','only_buys_gender_Children']

# kproto = KPrototypes(n_clusters=10, init='Cao', verbose=2)
# clusters_two = kproto.fit_predict(sample_sales_features[variable_set_three], categorical=[0,1,2,4,5,6])
# sample_sales_features['clust_2']=clusters_two
centroids_one=kproto.cluster_centroids_


# In[ ]:





# In[ ]:





# In[ ]:


sample_sales_features.iloc[:,1]

