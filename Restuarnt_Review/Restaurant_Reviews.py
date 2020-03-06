#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 10:28:04 2020

@author: amitkumar
"""

#Import the libraries
import numpy as np
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report

#Import the dataset
data = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t')
X = data['Review']
y = data['Liked']

#Checking the missing data if any
print(data.isnull().mean())

#reading stop words
stop_words = set(stopwords.words('english'))

#clean the data
corpus = []
for i in range(len(X)):
    review = re.sub('[^a-zA-Z]', ' ', X[i])
    review = review.lower()
    review = review.split()
    lemmatizer = WordNetLemmatizer()
    review = [lemmatizer.lemmatize(word) for word in review if not word in stop_words]
    review = ' '.join(review)
    corpus.append(review)
    
#Create the data frame of corpus
X = pd.DataFrame(data = corpus, columns = ['Review'])

#Split the data to train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

#Creating train and test corpus
train_corpus = [row for row in X_train['Review']]
test_corpus = [row for row in X_test['Review']]

#Applying TFIDF
#tfidf = TfidfVectorizer()
#tfidf.fit(X_train)
#X_train = tfidf.transform(train_corpus).toarray()
#X_test = tfidf.transform(test_corpus).toarray()

#Applying Bag of words model
vectorizer = CountVectorizer()
vectorizer.fit(train_corpus)
X_train = vectorizer.transform(train_corpus).toarray()
X_test = vectorizer.transform(test_corpus).toarray()

#Train the  Logistic Regression model
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

#Train the SVC model
#classifier = SVC(kernel='rbf')
#classifier.fit(X_train, y_train)



#Train the Random Forest Model
#classifier = RandomForestClassifier(n_estimators=200, criterion='entropy')
#classifier.fit(X_train, y_train)

#Predicting the test set result
y_pred = classifier.predict(X_test)

#Making classification report
print(classification_report(y_test, y_pred))
print(classification_report(y_train, classifier.predict(X_train)))

#Applying k-cross fold
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, 
                             scoring = 'accuracy', cv =10, n_jobs=-1)
accuracy = accuracies.mean()


