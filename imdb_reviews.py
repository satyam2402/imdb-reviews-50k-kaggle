
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


#importing the dataset

reviewdataset = pd.read_csv("/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv")


#Fetching top 5 records

reviewdataset.head()


#Importing ntlk
import nltk


#Download Stop words
nltk.download('stopwords')


#Importing regular expression
import re


#For removing stop words
from nltk.corpus import stopwords

#For the stem words
from nltk.stem.porter import PorterStemmer


#List of Important words in a review
reviewlist = []


#Replacing review characters apart from alphabets to space
for index in range(50000):
    review = reviewdataset['review'][index]
    # Removing html tags and non alphabetc characters
    review = re.sub('<.*?>',' ',review)
    review = re.sub('[^A-Za-z]',' ' ,review)
    
    #Turning review in lower case and spliting by space
    review = review.lower()
    review = review.split()
    
    #Collecting stem word if the word is not stopword
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    
    #Joining the review and appending the review in review list
    review = " ".join(review)
    reviewlist.append(review)


#Length of review list from which we choose max_features

len(reviewlist)


#Generating independant variables
#Bag of words model

from sklearn.feature_extraction.text import CountVectorizer
countvectorizer = CountVectorizer(max_features= 15000)

x = countvectorizer.fit_transform(reviewlist).toarray()


#Generating dependant variable 
y = (reviewdataset.iloc[:,-1].values == "positive").astype(int)


#Splitting the dataset into train and test
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x, y,test_size=0.20, random_state=0)


#Creating a classification model with bag of words
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(xtrain, ytrain)


#Getting prediction on test set
ypredict = classifier.predict(xtest)


#Accuracy retreival of model with confusion matrix
from sklearn.metrics import confusion_matrix
confusionmatrix = confusion_matrix(ytest, ypredict)


#Feching confusion metrix
confusionmatrix
