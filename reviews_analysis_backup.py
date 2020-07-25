#!/usr/bin/env python
# coding: utf-8

# Product Reviews Analysis
# Author- asethi
# Last updated - 07/08/2019

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

#importing packages
import nltk
from nltk import FreqDist
nltk.download('stopwords') # run this one time
nltk.download('punkt')
from sklearn.feature_extraction.text import TfidfTransformer
import codecs
from sklearn import feature_extraction
import datetime as dt
#export PYTHONWARNINGS="ignore"

from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer


import pandas as pd
#pd.set_option("display.max_colwidth", 200)
import numpy as np
import re
import spacy
import json
import os
import glob

import gensim
from gensim import corpora

# libraries for visualization
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import seaborn as sns
#get_ipython().run_line_magic('matplotlib', 'inline')
import en_core_web_sm
from nltk.corpus import stopwords
from configparser import SafeConfigParser, ConfigParser

from sqlalchemy import create_engine


from pandas.io.json import json_normalize
import pymysql
import sys
import en_core_web_sm
nlp=en_core_web_sm.load()


#df=pd.read_csv('Product_Reviews_Main+Sentiment.csv')





# Function to get source data from the source directory specified by configuration file
def get_source_data(config_file_name):

   parser = ConfigParser()
    parser.read(config_file_name)

    source_dir_path= parser.get('paths', 'source_dir')
    source_dir = os.listdir(source_dir_path )


    if(len(source_dir)):

        all_files = glob.glob(source_dir_path + "/*.csv")


        list = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)
            list.append(df)

        cust_review_data = pd.concat(list, axis=0, ignore_index=True)
        print(cust_review_data.shape)

    return cust_review_data






# Function to extract keywords using nlp

def keyword_extraction_main(df):


    df['product_review_content'] = df['product_review_content'].str.replace("[^a-zA-Z#]", " ")
    # remove short words (length < 3)
    df['product_review_content'] = df['product_review_content'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))

    # remove stopwords from the text
    reviews = [remove_stopwords(r.split()) for r in df['product_review_content']]

    # make entire text lowercase
    reviews = [r.lower() for r in reviews]

    
    nlp = en_core_web_sm.load()

    #bigrams=lambda row: [ngrams(reviews,2) for reviews in row]

    #nltk_tokens = nltk.word_tokenize(reviews) 

    tokenized_reviews = pd.Series(reviews).apply(lambda x: x.split())
    reviews2 = lemmatization(tokenized_reviews)


    reviews_3 = []
    for i in range(len(reviews2)):
        reviews_3.append(' '.join(reviews2[i]))

    df['reviews_keywords_content'] = reviews_3


    df['product_review_title'] = df['product_review_title'].str.replace("[^a-zA-Z#]", " ")
    # remove short words (length < 3)
    df['product_review_title'] = df['product_review_title'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))

    # remove stopwords from the text
    reviews_title = [remove_stopwords(r.split()) for r in df['product_review_title']]

    # make entire text lowercase
    reviews_title = [r.lower() for r in reviews_title]

    #import en_core_web_sm
    nlp = en_core_web_sm.load()

    #bigrams=lambda row: [ngrams(reviews,2) for reviews in row]

    #nltk_tokens = nltk.word_tokenize(reviews) 

    tokenized_reviews_title = pd.Series(reviews_title).apply(lambda x: x.split())
    reviews_title2 = lemmatization(tokenized_reviews_title)


    reviewstitle_3 = []
    for i in range(len(reviews_title2)):
        reviewstitle_3.append(' '.join(reviews_title2[i]))

    df['reviews_keywords_title'] = reviewstitle_3
    
    return df




# function to remove stopwords

def remove_stopwords(rev):

    stop_words = stopwords.words('english')

    rev_new = " ".join([i for i in rev if i not in stop_words])
    return rev_new

# function to plot most frequent terms for reference
def freq_words(x, terms = 30):
  all_words = ' '.join([text for text in x])
  all_words = all_words.split()

  fdist = FreqDist(all_words)
  words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

  # selecting top 20 most frequent words
  d = words_df.nlargest(columns="count", n = terms) 
  plt.figure(figsize=(20,5))
  ax = sns.barplot(data=d, x= "word", y = "count")
  ax.set(ylabel = 'Count')
  plt.show()



# Function to lemmatize the product review content
    
def lemmatization(texts,tags=['NOUN', 'ADJ','VERB','ADV']): # filter noun and adjective
       output = []
       for sent in texts:
             doc = nlp(" ".join(sent)) 
             output.append([token.lemma_ for token in doc if token.pos_ in tags])
      return output
                  

    
# Label sentiment based on extracted numerical values from ibm watson


def label_sentiment (data_reviews):

    data_reviews['sentiment_rating'] = np.where(data_reviews['sentiment']>0, 'Positive', 'Negative')
 
    return data_reviews



# Function to label purchase type into categories gifted or owned
        
def label_purchase_type(data_reviews,string_list_purchase_type):
    
    data_reviews['product_review_content'].str.lower()

    data_reviews['purchase_type']='Owned'
    data_reviews.reset_index(drop=True,inplace=True)
    #data_reviews['product_review_content'].to_lower
    for match_text in string_list_purchase_type:
        for j in range(len(data_reviews)):
            #print (j)
            if(match_text in data_reviews['product_review_content'][j]):
                data_reviews['purchase_type'][j]='Gift'

    return data_reviews




# Function to label defect type into Ripped Fabric, Holes in Fabric, Sewing Defect, Fraying Defect, Stich Issue, Durability Issue


def label_defect_type(data_reviews,string_list_defect_type):
    
    data_reviews['defect_type']='None-Reported'
    data_reviews.reset_index(drop=True,inplace=True)
    data_reviews['product_review_content'].str.lower()
    data_reviews['product_review_title'].str.lower()

    
    for match_text in string_list_defect_type:
        for j in range(len(data_reviews)):
            
            if(data_reviews['sentiment_rating'][j]=='Negative'):
                
                if(match_text in data_reviews['product_review_content'][j]):
                    
                    data_reviews['defect_type'][j]=string_list_defect_type.get(match_text)
                    
#                 if(match_text in data_reviews['product_review_title'][j]):
                    
#                     data_reviews['defect_type'][j]=match_text
       
    return data_reviews 




# Function to label value feedback into 'value worth' or 'not worth'

def label_value_feedback(data_reviews,value_feedback_list):
    
    data_reviews['value_label']='Value'
    data_reviews.reset_index(drop=True,inplace=True)
    data_reviews['product_review_content'].str.lower()
    data_reviews['product_review_title'].str.lower()

    
    for match_text in string_list_defect_type:
        for j in range(len(data_reviews)):
            
            #print(j)
            if((data_reviews['product_review_score'][j]) == 4 or (data_reviews['product_review_score'][j]) == 3 or (data_reviews['product_review_score'][j]) == 2 or (data_reviews['product_review_score'][j]) == 1):
                
                if(match_text in data_reviews['product_review_content'][j]):
                    
                    data_reviews['value_label'][j]='Expensive'
                    
#                 if(match_text in data_reviews['product_review_title'][j]):
                    
#                     data_reviews['value_label'][j]='Expensive'

    return data_reviews




# Function to label fit and feel of the product into the categories Fits Well,Rolls Down,Tight Fit,Fabric Stretchy,Loose Fit,size misfit,Fabric Itchy



def label_product_fit(data_reviews,fit_product_list):
    
    data_reviews['product_fit']='Fits Well'
    data_reviews.reset_index(drop=True,inplace=True)
    data_reviews['product_review_content'].str.lower()
    data_reviews['product_review_title'].str.lower()

    
    for match_text in fit_product_list:
        for j in range(len(data_reviews)):
            
            #print(j)
            if((data_reviews['product_review_score'][j]) == 4 or (data_reviews['product_review_score'][j]) == 3 or (data_reviews['product_review_score'][j]) == 2 or (data_reviews['product_review_score'][j]) == 1 or data_reviews['sentiment_rating'][j] == 'Negative' ):
                
                if(match_text in data_reviews['product_review_content'][j]):
                    
                    data_reviews['product_fit'][j]= fit_product_list.get(match_text)
                    
#                 if(match_text in data_reviews['product_review_title'][j]):
                    
#                     data_reviews['value_label'][j]='Expensive'

    return data_reviews


def label_product_feel(data_reviews,feel_product_list):
    
    data_reviews['product_feel']='None Issue Reported'
    data_reviews.reset_index(drop=True,inplace=True)
    data_reviews['product_review_content'].str.lower()
    data_reviews['product_review_title'].str.lower()

    
    for match_text in feel_product_list:
        for j in range(len(data_reviews)):
            
            #print(j)
            if((data_reviews['product_review_score'][j]) == 4 or (data_reviews['product_review_score'][j]) == 3 or (data_reviews['product_review_score'][j]) == 2 or (data_reviews['product_review_score'][j]) == 1 or data_reviews['sentiment_rating'][j] == 'Negative' ):
                
                if(match_text in data_reviews['product_review_content'][j]):
                    
                    data_reviews['product_feel'][j]= feel_product_list.get(match_text)
                    
#                 if(match_text in data_reviews['product_review_title'][j]):
                    
#                     data_reviews['value_label'][j]='Expensive'

    return data_reviews






# FUnction to label customer issues

def label_cust_issues(data_reviews,string_list_cust_issues):
    
    data_reviews['cust_issue']='Unclassified'
    data_reviews.reset_index(drop=True,inplace=True)
    data_reviews['product_review_content'].str.lower()
    data_reviews['product_review_title'].str.lower()

    
    for match_text in string_list_cust_issues:
        for j in range(len(data_reviews)):
            
            #print(j)
            if((data_reviews['product_review_score'][j]) == 4 or (data_reviews['product_review_score'][j]) == 3 or (data_reviews['product_review_score'][j]) == 2 or (data_reviews['product_review_score'][j]) == 1 or data_reviews['sentiment_rating'][j] == 'Negative' ):
                
                if(match_text in data_reviews['product_review_content'][j]):
                    
                    data_reviews['cust_issue'][j]= string_list_cust_issues.get(match_text)
                    
#                 if(match_text in data_reviews['product_review_title'][j]):
                    
#                     data_reviews['value_label'][j]='Expensive'

    return data_reviews



def overall_customer_feedback(data_reviews,string_list_cust_feedback):
    
    data_reviews['Overall_Feedback']='Unclassified'
    for row in range(len(data_reviews)):
        #print (row)
        for match_text in string_list_cust_feedback:
            #print(type(data_reviews['sentiment'][row]))
            if((data_reviews['sentiment'][row])>0):
                    data_reviews['Overall_Feedback'][row]='Positive'
            if((data_reviews['sentiment'][row])<0):
                    data_reviews['Overall_Feedback'][row]='Negative'
            if((data_reviews['sentiment'][row])>0.4 and match_text in data_reviews['product_review_title'][row]):
                    data_reviews['Overall_Feedback'][row]=string_list_cust_issues.get(match_text)
            if((data_reviews['sentiment'][row])<-0.4 and match_text in data_reviews['product_review_title'][row]):
                    data_reviews['Overall_Feedback'][row]=string_list_cust_issues.get(match_text)

                 
    return data_reviews

    







# Function to extract database parameters from the configuration files



def extract_database_params(config_file_name):

    config          = ConfigParser()
    config.read(config_file_name)

    db_params=dict()

    db_params['db_host']         = config.get('database', 'db_host')
    db_params['db_port']         = config.get('database', 'db_port')
    db_params['db_user']         = config.get('database', 'db_user')
    db_params['db_pass']         = config.get('database', 'db_pass')
    db_params['db_name']         = config.get('database', 'db_name')

    return db_params







def append_to_db(database_param_map,df_product_reviews_workset):

    try:

        # db credentials
        db_host = database_param_map['db_host']
        db_port = int(database_param_map['db_port'])
        db_user = database_param_map['db_user']
        db_pass = database_param_map['db_pass']
        db_name = database_param_map['db_name']

        # create connection
        connection = pymysql.connect(host=db_host, user=db_user, port=db_port,
                                     passwd=db_pass, db=db_name)
        cursor = connection.cursor(pymysql.cursors.DictCursor)

        # create engine
        engine = create_engine('mysql+pymysql://{a}:{b}@{c}/{d}'.format(a=db_user, b=db_pass, c=db_host, d=db_name))

        #print(extract_info.shap9)


        df_product_reviews_workset.to_sql('tb_cust_rev_analysis', con=engine, if_exists='append')

    except Exception as exc:

            exc_type, exc_obj, exc_tb = sys.exc_info()

            print(exc_type)
            print(exc_type)
            print(exc_obj)
            print(exc_tb.tb_lineno)
    finally:
            pass



# Deadling with source and processed files post processing


def file_post_processing(config_file_name,cust_review_data):

    parser = ConfigParser()
    parser.read(config_file_name)

    source_dir_path= parser.get('paths', 'source_dir')
    source_dir = os.listdir(source_dir_path )

    currentDT = dt.datetime.now()
    name_stamp=currentDT.strftime("%Y-%m-%d")
    processed_dir_path= parser.get('paths','processed_dir')
    processed_dir = os.listdir(processed_dir_path)
    processed_file='processed_data'+str(name_stamp)

    file_name = 'processed_data-'+str(dt.datetime.now().strftime("%Y-%m-%d"))
    cust_review_data.to_csv(os.path.join(processed_dir_path,file_name)+'.csv')

    #filelist = [ f for f in os.listdir(source_dir_path) if f.endswith(".csv") ]

    all_files = glob.glob(source_dir_path + "/*.csv")

    for f in all_files:
        os.remove(f)



##########################################################################################################
####################################################### Work Console #####################################
        
# Config file names         
config_file_name='nlp_params.ini'
label_config_file_name='label_params.ini'


df=get_source_data(config_file_name)
df_product_reviews_workset=keyword_extraction_main(df)


   
        
#Reading the custom strings from the configuration files for each of the files
        
parser = ConfigParser()
parser.read(label_config_file_name)
#########################
string_list_purchase_type=[]

selected_purchase=parser['purchase-type']

for key in selected_purchase:
    string_list_purchase_type.append(selected_purchase[key])

#########################
value_feedback_list=[]

value_feedback=parser['value_feedback_list']

for key in value_feedback:
   value_feedback_list.append(value_feedback[key])

#########################

fit_product_key=[]
fit_product_value=[]

fit_feel_selected=parser['fit_list']

for key in fit_feel_selected:
    fit_product_key.append(key)
    fit_product_value.append(fit_feel_selected[key])
    
fit_product_list=dict(zip(fit_product_key, fit_product_value))


##########################


feel_product_key=[]
feel_product_value=[]

product_feel_selected=parser['feel_list']

for key in product_feel_selected:
    feel_product_key.append(key)
    feel_product_value.append(product_feel_selected[key])
    
feel_product_list=dict(zip(fit_product_key, fit_product_value))


##############################



cust_issues_key=[]
cust_issues_value=[]

cust_issue_selected=parser['string_list_cust_issues']

for key in cust_issue_selected:
    cust_issues_key.append(key)
    cust_issues_value.append(cust_issue_selected[key])
    
string_list_cust_issues=dict(zip(cust_issues_key, cust_issues_value))

###############################

defect_type_key=[]
defect_type_value=[]

defect_type_selected=parser['defect_type']

#print(defect_type_selected)
for key in defect_type_selected:
    defect_type_key.append(key)
    defect_type_value.append(defect_type_selected[key])
        
string_list_defect_type=dict(zip(defect_type_key, defect_type_value))




cust_feedback_key=[]
cust_feedback_value=[]

cust_feedback_selected=parser['string_list_cust_feedback']


for key in cust_feedback_selected:
    cust_feedback_key.append(key)
    cust_feedback_value.append(cust_feedback_selected[key])
        
string_list_cust_feedback=dict(zip(defect_type_key, defect_type_value))


# Passing the functions created

df_product_reviews_workset.reset_index(drop=True)
df_product_reviews_workset=label_sentiment(df_product_reviews_workset)


df_product_reviews_workset=label_purchase_type(df_product_reviews_workset,string_list_purchase_type)

df_product_reviews_workset=label_defect_type(df_product_reviews_workset,string_list_defect_type)

df_product_reviews_workset=label_value_feedback(df_product_reviews_workset,value_feedback_list)

df_product_reviews_workset=label_product_fit(df_product_reviews_workset,fit_product_list)

df_product_reviews_workset=label_product_feel(df_product_reviews_workset,feel_product_list)

df_product_reviews_workset=label_cust_issues(df_product_reviews_workset,string_list_cust_issues)


database_param_map=extract_database_params(config_file_name)
print('Got DB params')

# Passing the function to append the data in database

append_to_db(database_param_map,df_product_reviews_workset)

# Invoking the function for post processing

file_post_processing(config_file_name,df)

print('success')

