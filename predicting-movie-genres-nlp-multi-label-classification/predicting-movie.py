import pandas as pd
import numpy as np
import json
import nltk
import re
import csv
import matplotlib.pyplot as plt 
import seaborn as sns
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

#%matplotlib inline
pd.set_option('display.max_colwidth', 300)


meta = pd.read_csv("MovieSummaries/movie.metadata.tsv", sep = '\t', 
                   header = None)
print("meta.head()",meta.head())

# rename columns
meta.columns = ["movie_id",1,"movie_name",3,4,5,6,7,"genre"]

plots = []

with open("MovieSummaries/plot_summaries.txt", 'r') as f:
       reader = csv.reader(f, dialect='excel-tab') 
       for row in tqdm(reader):
            plots.append(row)

movie_id = []
plot = []

# extract movie Ids and plot summaries
for i in tqdm(plots):
  movie_id.append(i[0])
  plot.append(i[1])

# create dataframe
movies = pd.DataFrame({'movie_id': movie_id, 'plot': plot})

print("movies.head()",movies.head())
                    
                    ############################################
                    #   Data Exploration and Pre-processing    #
                    ############################################
#Let’s add the movie names and their genres from the movie metadata file 
#by merging the latter into the former based on the movie_id column:

# change datatype of 'movie_id'
meta['movie_id'] = meta['movie_id'].astype(str)

# merge meta with movies
movies = pd.merge(movies, meta[['movie_id', 'movie_name', 'genre']], 
                  on = 'movie_id')

print("movies.head()",movies.head())
print("movies['genre'][0]",movies['genre'][0])

# an empty list
genres = [] 

# extract genres
for i in movies['genre']: 
  genres.append(list(json.loads(i).values())) 

# add to 'movies' dataframe  
movies['genre_new'] = genres

#Some of the samples might not contain any genre tags. We should remove those 
#samples as they won’t play a part in our model building process:

# remove samples with 0 genre tags
movies_new = movies[~(movies['genre_new'].str.len() == 0)]

print("movies_new.shape",movies_new.shape, "movies.shape", movies.shape)

#Notice that the genres are now in a list format. 
#Are you curious to find how many movie genres have been covered in this 
#dataset? The below code answers this question:

# get all genre tags in a list
all_genres = sum(genres,[])

print("len(set(all_genres)):", len(set(all_genres)))

# There are over 363 unique genre tags in our dataset. 
# That is quite a big number. I can hardy recall 5-6 genres! 
# Let’s find out what are these tags. We will use FreqDist( ) from the 
# nltk library to create a dictionary of genres and their occurrence count 
# across the dataset:

all_genres = nltk.FreqDist(all_genres) 

# create dataframe
all_genres_df = pd.DataFrame({'Genre': list(all_genres.keys()), 
                              'Count': list(all_genres.values())})

# I personally feel visualizing the data is a much better method than 
# simply putting out numbers. So, let’s plot the distribution of the 
# movie genres:
g = all_genres_df.nlargest(columns="Count", n = 50) 
plt.figure(figsize=(12,15)) 
ax = sns.barplot(data=g, x= "Count", y = "Genre") 
ax.set(ylabel = 'Count') 
plt.show()


# Next, we will clean our data a bit. I will use some very basic text cleaning
# steps (as that is not the focus area of this article):

# function for text cleaning 
def clean_text(text):
    # remove backslash-apostrophe 
    text = re.sub("\'", "", text) 
    # remove everything except alphabets 
    text = re.sub("[^a-zA-Z]"," ",text) 
    # remove whitespaces 
    text = ' '.join(text.split()) 
    # convert text to lowercase 
    text = text.lower() 
    
    return text

# Let’s apply the function on the movie plots by using the apply-lambda duo:
    
movies_new['clean_plot'] = movies_new['plot'].apply(lambda x: clean_text(x))

# In the clean_plot column, all the text is in lowercase and there are also no 
# punctuation marks. Our text cleaning has worked like a charm.

# The function below will visualize the words and their frequency in a set 
# of documents. Let’s use it to find out the most frequent words in the movie
# plots column:

def freq_words(x, terms = 30): 
  all_words = ' '.join([text for text in x]) 
  all_words = all_words.split() 
  fdist = nltk.FreqDist(all_words) 
  words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())}) 
  
  # selecting top 20 most frequent words 
  d = words_df.nlargest(columns="count", n = terms) 
  
  # visualize words and frequencies
  plt.figure(figsize=(12,15)) 
  ax = sns.barplot(data=d, x= "count", y = "word") 
  ax.set(ylabel = 'Word') 
  plt.show()
  
# print 100 most frequent words 
freq_words(movies_new['clean_plot'], 100)

# Most of the terms in the above plot are stopwords. These stopwords carry far
# less meaning than other keywords in the text (they just add noise to the 
#data). I’m going to go ahead and remove them from the plots’ text. 
#You can download the list of stopwords from the nltk library:

nltk.download('stopwords')

# Let’s remove the stopwords:
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# function to remove stopwords
def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)

movies_new['clean_plot'] = movies_new['clean_plot'].apply(lambda x: remove_stopwords(x))

#Check the most frequent terms sans the stopwords:

freq_words(movies_new['clean_plot'], 100)

                ###################################
                #   Converting Text to Features   #
                ###################################

# I mentioned earlier that we will treat this multi-label classification 
# problem as a Binary Relevance problem. Hence, we will now one hot encode 
# the target variable, i.e., genre_new by using sklearn’s 
# MultiLabelBinarizer( ). Since there are 363 unique genre tags, 
# there are going to be 363 new target variables.

from sklearn.preprocessing import MultiLabelBinarizer

multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(movies_new['genre_new'])

# transform target variable
y = multilabel_binarizer.transform(movies_new['genre_new'])

# Now, it’s time to turn our focus to extracting features from the cleaned 
# version of the movie plots data. For this article, I will be using TF-IDF 
# features. Feel free to use any other feature extraction method you are 
# comfortable with, such as Bag-of-Words, word2vec, GloVe, or ELMo.

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)

# I have used the 10,000 most frequent words in the data as my features. 
# You can try any other number as well for the max_features parameter.

# Now, before creating TF-IDF features, we will split our data into train and 
# validation sets for training and evaluating our model’s performance. 
# I’m going with a 80-20 split – 80% of the data samples in the train set and 
# the rest in the validation set:
    
# split dataset into training and validation set
xtrain, xval, ytrain, yval = train_test_split(movies_new['clean_plot'], y, test_size=0.2, random_state=9)

#Now we can create features for the train and the validation set:

# create TF-IDF features
xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
xval_tfidf = tfidf_vectorizer.transform(xval)

                ###############################################
                #   Build Your Movie Genre Prediction Model   # 
                ###############################################

# We are all set for the model building part! This is what we’ve been waiting 
# for.

# Remember, we will have to build a model for every one-hot encoded target 
# variable. Since we have 363 target variables, we will have to fit 
# 363 different models with the same set of predictors (TF-IDF features).

# As you can imagine, training 363 models can take a considerable amount of 
# time on a modest system. Hence, I will build a Logistic Regression model as 
# it is quick to train on limited computational power:

from sklearn.linear_model import LogisticRegression
# Binary Relevance
from sklearn.multiclass import OneVsRestClassifier
# Performance metric
from sklearn.metrics import f1_score

# We will use sk-learn’s OneVsRestClassifier class to solve this problem 
# as a Binary Relevance or one-vs-all problem:

lr = LogisticRegression()
clf = OneVsRestClassifier(lr)

# Finally, fit the model on the train set:
# fit model on train data
clf.fit(xtrain_tfidf, ytrain)
#Predict movie genres on the validation set:
# make predictions for validation set
y_pred = clf.predict(xval_tfidf)

#Let’s check out a sample from these predictions:
print(y_pred[3])

# It is a binary one-dimensional array of length 363. Basically, it is the 
# one-hot encoded form of the unique genre tags. We will have to find a way 
# to convert it into movie genre tags.

# Luckily, sk-learn comes to our rescue once again. We will use the 
# inverse_transform( ) function along with the MultiLabelBinarizer( ) 
# object to convert the predicted arrays into movie genre tags:

print(multilabel_binarizer.inverse_transform(y_pred)[3])

# evaluate performance
f1_score = f1_score(yval, y_pred, average="micro")

                    #################################
                    #   Create Inference Function   #
                    #################################

# Wait – we are not done with the problem yet. We also have to take care of 
# the new data or new movie plots that will come in the future, right? Our 
# movie genre prediction system should be able to take a movie plot in raw 
# form as input and generate its genre tag(s).

# To achieve this, let’s build an inference function. It will take a movie 
# plot text and follow the below steps:

# * Remove stopwords from the cleaned text
# * Extract features from the text
# * Make predictions
# * Return the predicted movie genre tags

def infer_tags(q):
    q = clean_text(q)
    q = remove_stopwords(q)
    q_vec = tfidf_vectorizer.transform([q])
    q_pred = clf.predict(q_vec)
    return multilabel_binarizer.inverse_transform(q_pred)

# Let’s test this inference function on a few samples from our validation set:
for i in range(5): 
  k = xval.sample(1).index[0] 
  print("Movie: ", movies_new['movie_name'][k], 
        "\nPredicted genre: ", infer_tags(xval[k]))
  print("Actual genre: ",movies_new['genre_new'][k], "\n")





