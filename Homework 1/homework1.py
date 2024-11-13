#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Task 1: simple predictor that estimates rating from review length

"""
star rating ≃ θ0 + θ1 × [review length in characters].
scale the feature to be between 0 and 1 by dividing by the maximum review length in the dataset. 
report the values θ0 and θ1, and the Mean Squared Error of your predictor (on the entire dataset).

"""

import json
import gzip
import numpy as np

# Load the dataset
f = gzip.open("fantasy_10000.json.gz")
dataset = []
for l in f:
    dataset.append(json.loads(l))

# Extract review lengths and ratings
review_lengths = [len(d['review_text']) for d in dataset]
ratings = [d['rating'] for d in dataset]

# Scale review lengths between 0 and 1
max_length = max(review_lengths)
scaled_lengths = [length / max_length for length in review_lengths]

# Create a feature matrix
X = np.array(scaled_lengths).reshape(-1, 1)
y = np.array(ratings)

# Fit a linear regression model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)

# Get the coefficients
theta0 = model.intercept_
theta1 = model.coef_[0]

# Calculate Mean Squared Error (MSE)
from sklearn.metrics import mean_squared_error
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)

print(f"Theta0: {theta0}")
print(f"Theta1: {theta1}")
print(f"Mean Squared Error: {mse}")


# In[2]:


# Task 2: include (in addition to the scaled length) features based on the time of the review

"""
include features based on the time of the review.
use a one-hot encoding for the weekday and month.
write down feature vectors for the first two examples.
"""

import json
import gzip
import numpy as np
import dateutil.parser
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
f = gzip.open("fantasy_10000.json.gz")
dataset = []
for l in f:
    dataset.append(json.loads(l))
    
# Extract review lengths, ratings and days
review_lengths = np.array([len(d['review_text']) for d in dataset]).reshape(-1,1)
ratings = np.array([d['rating'] for d in dataset])
dates = np.array([d['date_added'] for d in dataset])

# Scale the review_lengths to be between 0 and 1
max_length = max(review_lengths)
scaled_lengths = review_lengths / max_length

# Extract weekday and month from the date strings
weekdays = [dateutil.parser.parse(d).weekday() for d in dates]
months = [dateutil.parser.parse(d).month for d in dates]

# One-hot encoding for weekdays and months, dropping the first dimension for both
encoder = OneHotEncoder(categories=[list(range(7)), list(range(1, 13))], drop=[0, 1], sparse=False)
encoded_features = encoder.fit_transform(np.array([weekdays, months]).T)

# Create an offset term (a column of ones)
offsets = np.ones((len(review_lengths), 1))

# Combine intercept, scaled_lengths, and encoded_features
X = np.hstack((offsets, scaled_lengths, encoded_features))

# Feature vectors for the first two examples
print("Feature vector for the first example:", list(X[0]))
print("Feature vector for the second example:", list(X[1]))


# In[3]:


# Task 3: use the weekday and month values directly

"""
use the weekday and month values directly as features.
use the one-hot encoding from Question 2.
report the MSE of each.
"""

########################## Use Weekday and Month as One-Hot ##########################

import json
import gzip
import dateutil.parser
import numpy as np
from sklearn.linear_model import LinearRegression

# Convert lists to numpy arrays
y = np.array(ratings)

# Train a linear regression model
model = LinearRegression()
model.fit(X, y)

# Calculate Mean Squared Error (MSE)
from sklearn.metrics import mean_squared_error
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
print(f"Mean Squared Error (Use one-hot encoding for the weekday and month): {mse}")

########################## Use Weekday and Month Directly ##########################

import json
import gzip
import dateutil.parser
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the dataset
f = gzip.open("fantasy_10000.json.gz")
dataset = []
for l in f:
    dataset.append(json.loads(l))

# Initialize lists to store features and labels
features = []  # Feature vectors
ratings = []  # Ratings

# Iterate through the dataset and extract features
for d in dataset:
    # Parse the review date
    t = dateutil.parser.parse(d['date_added'])
    
    # Extract weekday and month
    weekday = t.weekday()
    month = t.month
    
    # Calculate the length of the review text and scale it
    review_length = len(d['review_text'])
    scaled_review_length = review_length / max_length
    
    # Create the feature vector by combining all features
    feature_vector = np.array(([scaled_review_length, weekday, month]))
    
    # Append the feature vector and rating to the lists
    features.append(feature_vector)
    ratings.append(d['rating'])

# Convert lists to numpy arrays
X = np.array(features)
y = np.array(ratings)

# Train a linear regression model
model = LinearRegression()
model.fit(X, y)

# Calculate Mean Squared Error (MSE)
from sklearn.metrics import mean_squared_error
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
print(f"Mean Squared Error (Use the weekday and month Directly): {mse}")


# In[24]:


# Task 4: split the data into 50%/50% train/test

"""
split the data into 50%/50% train/test fractions.
report the MSE of the two models on the test set.
"""

########################## Use Weekday and Month as One-Hot ##########################

import json
import gzip
import numpy as np
import dateutil.parser
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
f = gzip.open("fantasy_10000.json.gz")
dataset = []
for l in f:
    dataset.append(json.loads(l))

random.seed(0)
random.shuffle(dataset)
    
# Extract review lengths, ratings and days
review_lengths = np.array([len(d['review_text']) for d in dataset]).reshape(-1,1)
ratings = np.array([d['rating'] for d in dataset])
dates = np.array([d['date_added'] for d in dataset])

# Scale the review_lengths to be between 0 and 1
max_length = max(review_lengths)
scaled_lengths = review_lengths / max_length

# Extract weekday and month from the date strings
weekdays = [dateutil.parser.parse(d).weekday() for d in dates]
months = [dateutil.parser.parse(d).month for d in dates]

# One-hot encoding for weekdays and months, dropping the first dimension for both
encoder = OneHotEncoder(categories=[list(range(7)), list(range(1, 13))], drop=[0, 1], sparse=False)
encoded_features = encoder.fit_transform(np.array([weekdays, months]).T)

# Create an offset term (a column of ones)
offsets = np.ones((len(review_lengths), 1))

# Combine intercept, scaled_lengths, and encoded_features
X = np.hstack((offsets, scaled_lengths, encoded_features))
y = np.array(ratings)

from sklearn.model_selection import train_test_split

# Split the data into training and test sets (50% each)
X_train, X_test = X[:len(X)//2], X[len(X)//2:]
y_train, y_test = y[:len(y)//2], y[len(y)//2:]

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Calculate Mean Squared Error (MSE)
from sklearn.metrics import mean_squared_error
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (Use the weekday and month Directly) on the test set: {mse}")

########################## Use Weekday and Month Directly ##########################

import json
import gzip
import dateutil.parser
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the dataset
f = gzip.open("fantasy_10000.json.gz")
dataset = []
for l in f:
    dataset.append(json.loads(l))
    
random.seed(0)
random.shuffle(dataset)

# Initialize lists to store features and labels
features = []  # Feature vectors
ratings = []  # Ratings

# Iterate through the dataset and extract features
for d in dataset:
    # Parse the review date
    t = dateutil.parser.parse(d['date_added'])
    
    # Extract weekday and month
    weekday = t.weekday()
    month = t.month
    
    # Calculate the length of the review text and scale it
    review_length = len(d['review_text'])
    scaled_review_length = review_length / max_length
    
    # Create the feature vector by combining all features
    feature_vector = np.array(([scaled_review_length, weekday, month]))
    
    # Append the feature vector and rating to the lists
    features.append(feature_vector)
    ratings.append(d['rating'])

# Convert lists to numpy arrays
X = np.array(features)
y = np.array(ratings)

from sklearn.model_selection import train_test_split

# Split the data into training and test sets (50% each)
X_train, X_test = X[:len(X)//2], X[len(X)//2:]
y_train, y_test = y[:len(y)//2], y[len(y)//2:]

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Calculate Mean Squared Error (MSE)
from sklearn.metrics import mean_squared_error
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (Use the weekday and month Directly) on the test set: {mse}")


# In[8]:


# Task 5: fit a logistic regressor

"""
fit a logistic regressor that estimates the binarized score from review length.
use the class weight=’balanced’ option, report the number of True Positives, 
True Negatives, False Positives, False Negatives, and the Balanced Error Rate of the classifier.
"""

import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

# Load the data from "beer_50000.json"
f = open("beer_50000.json")
dataset = []
for l in f:
    dataset.append(eval(l))

# Create a label vector (1 for positive, 0 for negative) based on review scores
labels = np.array([1 if data['review/overall'] >= 4 else 0 for data in dataset]).reshape(-1,1)

# Extract the review lengths
review_lengths = np.array([len(data['review/text']) for data in dataset]).reshape(-1,1)

# Fit a logistic regression model with class_weight='balanced'
model = LogisticRegression(class_weight='balanced')
model.fit(review_lengths, labels)

# Predict the labels on the dataset
pred = model.predict(review_lengths)

# Calculate the confusion matrix
tn, fp, fn, tp = confusion_matrix(labels, pred).ravel()

# Calculate the Balanced Error Rate (BER)
ber = 1 - balanced_accuracy_score(labels, pred)

# Print the results
print("True Positives:", tp)
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("Balanced Error Rate (BER):", ber)


# In[37]:


# Task 6: compute the precision of classifier

"""
compute the precision@K of classifier for K ∈ {1, 100, 1000, 10000}.
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score

# Load the data from "beer_50000.json"
f = open("beer_50000.json")
dataset = []
for l in f:
    dataset.append(eval(l))

# Construct the label vector
y = np.array([d['review/overall'] >= 4 for d in dataset])

# Extract the review length
X = np.array([len(d['review/text']) for d in dataset]).reshape(-1, 1)

# Fit a logistic regressor
log_reg = LogisticRegression(class_weight='balanced')
log_reg.fit(X, y)

# Compute precision at K
K_values = [1, 100, 1000, 10000]
for k in K_values:
    confidence_scores = log_reg.decision_function(X)
    predicted_ranking = np.argsort(confidence_scores)[::-1][:k]
    y_pred_at_k = np.zeros(len(y))
    y_pred_at_k[predicted_ranking] = 1
    precision_at_k = precision_score(y, y_pred_at_k)
    print(f"Precision@{k}: {precision_at_k}")


# In[26]:


# Task 7: improve the classifier

"""
reduce the balanced error rate by incorporating additional features from the data.
describe your improvement (as a string) and report the BER of your new predictor.
"""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

# Load the data from "beer_50000.json"
f = open("beer_50000.json")
dataset = []
for l in f:
    dataset.append(eval(l))

# Create a label vector (1 for positive, 0 for negative) based on review scores
labels = [1 if data['review/overall'] >= 4 else 0 for data in dataset]

# Extract additional features: 'beer/style', ratings, and review text
beer_styles = [data['beer/style'] for data in dataset]
ratings = np.array([[data['review/appearance'], data['review/aroma'], data['review/palate'], data['review/taste']] for data in dataset])
review_text = [data['review/text'] for data in dataset]

# Encode beer styles using one-hot encoding
encoder = OneHotEncoder(sparse=False)
beer_styles_encoded = encoder.fit_transform(np.array(beer_styles).reshape(-1, 1))

# Scale ratings using StandardScaler
scaler = StandardScaler()
ratings_scaled = scaler.fit_transform(ratings)

# Vectorize the review text using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # Adjust the number of features as needed
review_text_tfidf = tfidf_vectorizer.fit_transform(review_text)

# Combine all the features into one feature matrix
X = hstack((beer_styles_encoded, ratings_scaled, review_text_tfidf))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)

# Fit a logistic regression model with class_weight='balanced'
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

# Predict the labels on the test set
y_pred = model.predict(X_test)

# Calculate the Balanced Error Rate (BER)
ber = 1 - balanced_accuracy_score(y_test, y_pred)

# Print the results
print("Balanced Error Rate (BER):", ber)


# In[25]:


# Answer Template
import json
from collections import defaultdict
from sklearn import linear_model
import numpy
import random
import gzip
import dateutil.parser
import math

answers = {}

def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[26]:


### Question 1

answers['Q1'] = [3.685681355016952, 0.983353918106614, 1.5522086622355378]
assertFloatList(answers['Q1'], 3)


# In[27]:


### Question 2

answers['Q2'] = [[1,0.14581294561722355,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0], 
                 [1,0.10631902698168601,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]]
assertFloatList(answers['Q2'][0], 19)
assertFloatList(answers['Q2'][1], 19)


# In[28]:


### Question 3

answers['Q3'] = [1.5467978637695312, 1.5516353711453328]
assertFloatList(answers['Q3'], 2)


# In[29]:


### Question 4

answers['Q4'] = [1.6264453676167938, 1.6282919476176059]
assertFloatList(answers['Q4'], 2)


# In[30]:


### Question 5

answers['Q5'] = [14201, 10503, 5885, 19411, 0.4683031525957275]
assertFloatList(answers['Q5'], 5)


# In[38]:


### Question 6

answers['Q6'] = [1.0,0.75,0.71,0.7146]
assertFloatList(answers['Q6'], 4)


# In[39]:


### Question 7
its_test_BER = 0.16405903360796215
answers['Q7'] = ["Add beer styles, ratings, features from text", its_test_BER]


# In[40]:


f = open("answers_hw1.txt", 'w')
f.write(str(answers) + '\n')
f.close()

