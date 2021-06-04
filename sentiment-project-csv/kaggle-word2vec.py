# Firstly, please note that the performance of google word2vec is better on big datasets. 
# In this example we are considering only 25000 training examples from the imdb dataset.
# Therefore, the performance is similar to the "bag of words" model.

# Importing libraries
import numpy as np
import pandas as pd
# BeautifulSoup is used to remove html tags from the text
from bs4 import BeautifulSoup 
import re # For regular expressions
import sklearn
from sklearn import metrics




# Stopwords can be useful to undersand the semantics of the sentence.
# Therefore stopwords are not removed while creating the word2vec model.
# But they will be removed  while averaging feature vectors.
from nltk.corpus import stopwords

# Read data from files
train = pd.read_csv("filename-train.csv")

# train_text = np.asarray(train['text'])
# print(train_text)
# train_text = train_text[np.logical_not(np.isnan(train_text))]
# train_text = np.nan_to_num(train_text)

test = pd.read_csv("filename-validation.csv")

# This function converts a text to a sequence of words.
def review_wordlist(review, remove_stopwords=False):
    # 1. Removing html tags
    #review_text = BeautifulSoup(str(review)).get_text()
    review_text = review
    # 2. Removing non-letter.
    review_text = re.sub("[^a-zA-Z]"," ",review_text)
    # 3. Converting to lower case and splitting
    words = review_text.lower().split()
    # 4. Optionally remove stopwords
    if remove_stopwords:
        stops = set(stopwords.words("english"))     
        words = [w for w in words if not w in stops]
    
    return(words)


# word2vec expects a list of lists.
# Using punkt tokenizer for better splitting of a paragraph into sentences.

import nltk.data
#nltk.download('popular')

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# This function splits a review into sentences
def review_sentences(review, tokenizer, remove_stopwords=False):
    # 1. Using nltk tokenizer
    #print(review)
    raw_sentences = tokenizer.tokenize(str(review).strip())
    #raw_sentences = tokenizer.tokenize(review)
    sentences = []
    # 2. Loop for each sentence
    for raw_sentence in raw_sentences:
        if len(raw_sentence)>0:
            sentences.append(review_wordlist(raw_sentence,\
                                            remove_stopwords))

    # This returns the list of lists
    return sentences


sentences = []
print("Parsing sentences from training set")
for review in train["text"]:
   # print(review)
    sentences += review_sentences(review, tokenizer)
    
# Importing the built-in logging module
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Creating the model and setting values for the various parameters
num_features = 300  # Word vector dimensionality
min_word_count = 1 # Minimum word count
num_workers = 4     # Number of parallel threads
context = 10        # Context window size
downsampling = 1e-3 # (0.001) Downsample setting for frequent words

# Initializing the train model
from gensim.models import word2vec
print("Training model....")
model = word2vec.Word2Vec(sentences,\
                          workers=num_workers,\
                          vector_size=num_features,\
                          min_count=min_word_count,\
                          window=context,
                          sample=downsampling)

# To make the model memory efficient
model.init_sims(replace=True)

# Saving the model for later use. Can be loaded using Word2Vec.load()
model_name = "300features_40minwords_10context"
model.save(model_name)

# Few tests: This will print the odd word among them 
model.wv.doesnt_match("man woman dog child kitchen".split())


# Function to average all word vectors in a paragraph
def featureVecMethod(words, model, num_features):
    # Pre-initialising empty numpy array for speed
    featureVec = np.zeros(num_features,dtype="float32")
    nwords = 0
    
    #Converting Index2Word which is a list to a set for better speed in the execution.
    index2word_set = set(model.wv.index_to_key)
    
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec,model.wv[word])
    
    # Dividing the result by number of words to get average
    featureVec = np.divide(featureVec, nwords)
    return featureVec

# Function for calculating the average feature vector
def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    for review in reviews:
        # Printing a status message every 1000th review
        if counter%1000 == 0:
            print("Review %d of %d"%(counter,len(reviews)))
            
        reviewFeatureVecs[counter] = featureVecMethod(review, model, num_features)
        counter = counter+1
        
    return reviewFeatureVecs
# Calculating average feature vector for training set
clean_train_reviews = []
for review in train['text']:
    clean_train_reviews.append(review_wordlist(review, remove_stopwords=True))
    
trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)
# Calculating average feature vactors for test set     
clean_test_reviews = []
for review in test["text"]:
    clean_test_reviews.append(review_wordlist(review,remove_stopwords=True))
    
testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)
# Fitting a random forest classifier to the training data
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 100)
    
print("Fitting random forest to training data....")  
print("TRAIN STAR PRINTED HERE")
#print(type(train['stars'][0]))
train_star = train['stars'].astype(np.float32)
train_star_new = []
for star in train_star:
    #print(star)
    if star == float("Nan"):
        print("HELPWITHCS221")
        train_star_new.append(0)
    else:
        train_star_new.append(star)
# train_star = np.asarray(train['stars'])
# train_star = train_star[np.logical_not(np.isnan(train_star))]
# train_star = np.nan_to_num(train_star)


#train_star = np.nan_to_num(np.where(np.isnan(train['stars'])))
forest = forest.fit(trainDataVecs, train_star_new) #previously sentiment

# Predicting the sentiment values for test data and saving the results in a csv file 
result = forest.predict(testDataVecs)
#print('Mean Absolute Error:', np.sqrt(metrics.mean_absolute_error(result, test['stars'])))
print('Misclassification Error:', 1-metrics.accuracy_score(result, test['stars'], normalize=True))

preds = result
print(sklearn.metrics.f1_score(test['stars'], preds, average='binary', sample_weight=None, zero_division='warn'))
output = pd.DataFrame(data={"text":test["text"], "stars":result})
output.to_csv( "output.csv")