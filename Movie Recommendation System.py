#!/usr/bin/env python
# coding: utf-8

# #  Importing the dependencies

# In[29]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib


# # Data Collection and pre-processing

# In[7]:


#load data
movie_data = pd.read_csv('D:\ML Files\Movie Recommendation System\movies.csv')


# In[8]:


movie_data.head()


# In[9]:


movie_data.shape


# In[10]:


movie_data.isnull().sum()


# In[14]:


#selecting the features for recommendation
selected_features = ['genres','keywords','tagline','cast','director']
print(selected_features)


# In[15]:


#Replacing Null Values
for feature in selected_features:
    movie_data[feature] = movie_data[feature].fillna('')


# In[16]:


#Feacture combined
combined_data = movie_data['genres']+" "+movie_data['keywords']+" "+movie_data['tagline']+" "+movie_data['cast']+" "+movie_data['director']


# In[17]:


print(combined_data)


# In[18]:


#Vectorizing
vectorizer = TfidfVectorizer()


# In[21]:


feature_vector = vectorizer.fit_transform(combined_data)
print(feature_vector)


# # Cosine Similarity

# In[23]:


#Getting the similarity score using cosine similarities
similarity = cosine_similarity(feature_vector)


# In[24]:


print(similarity)


# In[25]:


print(similarity.shape)


# In[26]:


#Getting the movie name from the user

movie_name = input("Enter watched movie name : ")


# In[27]:


#List of movie names in dataset
list_names = movie_data['title'].tolist()
print(list_names)


# In[31]:


#Finding the best match for the given movie name

close_match = difflib.get_close_matches(movie_name,list_names)[0]
print(close_match)


# In[33]:


#Index of given movie
index_of_movie = movie_data[movie_data.title == close_match]['index'].values[0]
print(index_of_movie)


# In[34]:


#Getting similar movies

similar_score = list(enumerate(similarity[index_of_movie]))
print(similar_score)


# In[36]:


#Sort similar movies

sorted_movies = sorted(similar_score,key = lambda x:x[1],reverse=True)


# In[37]:


#Print first 10 movies
j=0
print("Suggested Movies -------------------------->>>> ")
for i in sorted_movies:
    index = i[0]
    title_of_movies = movie_data[movie_data.index == index]['title'].values[0]
    if(j<10):
        print(j,". ",title_of_movies)
        j+=1


# In[ ]:




