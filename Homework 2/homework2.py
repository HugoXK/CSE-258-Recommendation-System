#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Task 1: use the style of the beer to predict its ABV

"""
construct a one-hot encoding of the beer style, for those categories that appear in more than 1,000 reviews.
train a logistic regressor using this one-hot encoding to predict whether beers have an ABV greater than 7 percent. 
train the classifier on the training set.
report its performance in terms of the accuracy and Balanced Error Rate (BER) on the validation and test sets.
use a regularization constant of C = 10.
"""

import random
from sklearn import linear_model
from matplotlib import pyplot as plt
from collections import defaultdict
import gzip
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from collections import defaultdict
import json

# Function to parse the data from the file
def parseData(fname):
    for l in open(fname):
        yield eval(l)

# Loading and shuffling the data
data = list(parseData("beer_50000.json"))
random.seed(0)
random.shuffle(data)

# Splitting the data into train, validation, and test sets
dataTrain = data[:25000]
dataValid = data[25000:37500]
dataTest = data[37500:]

# Creating target labels for the data sets
yTrain = [d['beer/ABV'] > 7 for d in dataTrain]
yValid = [d['beer/ABV'] > 7 for d in dataValid]
yTest = [d['beer/ABV'] > 7 for d in dataTest]

# Calculating category counts for beer styles
categoryCounts = defaultdict(int)
for d in data:
    categoryCounts[d['beer/style']] += 1
# Filtering categories based on counts
categories = [c for c in categoryCounts if categoryCounts[c] > 1000]
# Creating a mapping of categories to feature indices
catID = dict(zip(list(categories), range(len(categories))))

# One-hot encoding function
def feature(datum):
    feat = [0] * len(categories)
    if datum['beer/style'] in categories:
        feat[catID[datum['beer/style']]] = 1
    return feat

# Applying one-hot encoding to the data sets
XTrain = [feature(d) for d in dataTrain]
XValid = [feature(d) for d in dataValid]
XTest = [feature(d) for d in dataTest]

# Logistic Regression model training
logreg = linear_model.LogisticRegression(C=10, class_weight='balanced')
logreg.fit(XTrain, yTrain)

# Making predictions on the validation and test sets
yValidPred = logreg.predict(XValid)
yTestPred = logreg.predict(XTest)

# Computing performance metrics
valid_accuracy = accuracy_score(yValid, yValidPred)
valid_ber = 1 - balanced_accuracy_score(yValid, yValidPred)

test_accuracy = accuracy_score(yTest, yTestPred)
test_ber = 1 - balanced_accuracy_score(yTest, yTestPred)

# Printing the performance metrics
print(f'Validation Set - Accuracy: {valid_accuracy}, BER: {valid_ber}')
print(f'Test Set - Accuracy: {test_accuracy}, BER: {test_ber}')


# In[2]:


# Task 2: extend model to include more features

"""
extend the model to include a vector of five ratings and the review length (in characters). 
scale the ‘length’ feature to be between 0 and 1 by dividing by the maximum length seen during training. 
use C = 10 and report the validation and test BER of the new classifier.
"""

# Function to parse the data from the file
def parseData(fname):
    for l in open(fname):
        yield eval(l)

# Loading and shuffling the data
data = list(parseData("beer_50000.json"))
random.seed(0)
random.shuffle(data)

# Splitting the data into train, validation, and test sets
dataTrain = data[:25000]
dataValid = data[25000:37500]
dataTest = data[37500:]

# Creating target labels for the data sets
yTrain = [d['beer/ABV'] > 7 for d in dataTrain]
yValid = [d['beer/ABV'] > 7 for d in dataValid]
yTest = [d['beer/ABV'] > 7 for d in dataTest]

# Calculating category counts for beer styles
categoryCounts = defaultdict(int)
for d in data:
    categoryCounts[d['beer/style']] += 1
# Filtering categories based on counts
categories = [c for c in categoryCounts if categoryCounts[c] > 1000]
# Creating a mapping of categories to feature indices
catID = dict(zip(list(categories), range(len(categories))))

# Function to extract features from the data
def feature(datum, max_length):
    # Create one-hot encoded features for beer styles
    feat = [0] * len(categories)
    if datum['beer/style'] in categories:
        feat[catID[datum['beer/style']]] = 1

    # Extract the review ratings
    ratings = [datum['review/aroma'], datum['review/appearance'], datum['review/palate'], datum['review/taste'], datum['review/overall']]

    # Extract the review length
    length = len(datum['review/text'])

    # Scale the 'length' feature to be between 0 and 1
    length_scaled = length / max_length

    # Combine all features into a single list
    feat += ratings + [length_scaled]
    
    return feat

# Computing the maximum length of reviews in the training set
max_length_train = max(len(datum['review/text']) for datum in dataTrain)

# Applying the extended features to the data sets
XTrain = [feature(d, max_length_train) for d in dataTrain]
XValid = [feature(d, max_length_train) for d in dataValid]
XTest = [feature(d, max_length_train) for d in dataTest]

# Logistic Regression model training with extended features
logreg_extended = linear_model.LogisticRegression(C=10, class_weight='balanced')
logreg_extended.fit(XTrain, yTrain)

# Making predictions on the validation and test sets using the extended model
yValidPred_extended = logreg_extended.predict(XValid)
yTestPred_extended = logreg_extended.predict(XTest)

# Computing performance metrics for the extended model
valid_ber_extended = 1 - balanced_accuracy_score(yValid, yValidPred_extended)
test_ber_extended = 1 - balanced_accuracy_score(yTest, yTestPred_extended)

# Printing the validation and test BER for the extended model
print(f'Extended Model - Validation BER: {valid_ber_extended}')
print(f'Extended Model - Test BER: {test_ber_extended}')


# In[3]:


# Task 3: implement a complete regularization pipeline with the balanced classifier

"""
split your data from above in half so that you have 50%/25%/25% train/validation/test fractions. 
consider values of C in the range {0.001, 0.01, 0.1, 1, 10}. 
report the validation BER for each value of C. 
report which value of C you would ultimately select for your model, and that model’s performance on the validation and test sets.
"""

# Splitting data into 50%/25%/25% train/validation/test fractions
dataTrain, dataOthers = train_test_split(data, test_size=0.5, random_state=0)
dataValid, dataTest = train_test_split(dataOthers, test_size=0.5, random_state=0)

# Extracting target labels for the new data splits
yTrain = [d['beer/ABV'] > 7 for d in dataTrain]
yValid = [d['beer/ABV'] > 7 for d in dataValid]
yTest = [d['beer/ABV'] > 7 for d in dataTest]

# Computing the maximum length of reviews in the new training set
max_length_train = max(len(datum['review/text']) for datum in dataTrain)

# Defining the range of C values to consider
C_values = [0.001, 0.01, 0.1, 1, 10]

# Dictionary to store the validation BER for each value of C
validation_ber_dict = {}

# Training and evaluating the model for each value of C
for C in C_values:
    # Feature extraction with extended features and scaling
    XTrain = [feature(d, max_length_train) for d in dataTrain]
    XValid = [feature(d, max_length_train) for d in dataValid]

    # Logistic Regression model training with current C value
    logreg_regularized = linear_model.LogisticRegression(C=C, class_weight='balanced')
    logreg_regularized.fit(XTrain, yTrain)

    # Making predictions on the validation set
    yValidPred_regularized = logreg_regularized.predict(XValid)

    # Computing and storing the validation BER for the current C value
    validation_ber = 1 - balanced_accuracy_score(yValid, yValidPred_regularized)
    validation_ber_dict[C] = validation_ber

# Finding the value of C that gives the lowest validation BER
best_C = min(validation_ber_dict, key=validation_ber_dict.get)

# Retrain the model on the combined train and validation sets using the best C value
XTrainFull = [feature(d, max_length_train) for d in dataTrain + dataValid]
yTrainFull = yTrain + yValid
logreg_best = linear_model.LogisticRegression(C=best_C, class_weight='balanced')
logreg_best.fit(XTrainFull, yTrainFull)

# Evaluate the model's performance on the validation and test sets using the best C value
yValidPred_best = logreg_best.predict([feature(d, max_length_train) for d in dataValid])
yTestPred_best = logreg_best.predict([feature(d, max_length_train) for d in dataTest])
validation_ber_best = 1 - balanced_accuracy_score(yValid, yValidPred_best)
test_ber_best = 1 - balanced_accuracy_score(yTest, yTestPred_best)

# Printing the validation BER for each value of C and the best C value
print("Validation BER for each value of C:")
for C, val_ber in validation_ber_dict.items():
    print(f"C={C}: Validation BER={val_ber}")
print(f"Best C value: {best_C}")

# Printing the model's performance on the validation and test sets using the best C value
print(f"Model performance on the validation set (C={best_C}): Validation BER={validation_ber_best}")
print(f"Model performance on the test set (C={best_C}): Test BER={test_ber_best}")


# In[4]:


# Task 4: An ablation study

"""
measure the marginal benefit of various features by re-training the model with one feature ‘ablated’ at a time. 
consider each of the three features in your classifier above, and setting C = 1.
report the test BER with only the other two features and the third deleted.
"""

# Function to parse the data from the file
def parseData(fname):
    for l in open(fname):
        yield eval(l)

# Loading and shuffling the data
data = list(parseData("beer_50000.json"))
random.seed(0)
random.shuffle(data)

# Splitting the data into train, validation, and test sets
dataTrain = data[:25000]
dataValid = data[25000:37500]
dataTest = data[37500:]

# Creating target labels for the data sets
yTrain = [d['beer/ABV'] > 7 for d in dataTrain]
yValid = [d['beer/ABV'] > 7 for d in dataValid]
yTest = [d['beer/ABV'] > 7 for d in dataTest]

# Calculating category counts for beer styles
categoryCounts = defaultdict(int)
for d in data:
    categoryCounts[d['beer/style']] += 1
# Filtering categories based on counts
categories = [c for c in categoryCounts if categoryCounts[c] > 1000]
# Creating a mapping of categories to feature indices
catID = dict(zip(list(categories), range(len(categories))))

from sklearn.metrics import balanced_accuracy_score

# Ablation study for 'beer style' feature
def feature_without_style(datum, max_length):
    # Extract the review ratings
    ratings = [datum['review/aroma'], datum['review/appearance'], datum['review/palate'], datum['review/taste'], datum['review/overall']]

    # Extract the review length
    length = len(datum['review/text'])

    # Scale the 'length' feature to be between 0 and 1
    length_scaled = length / max_length

    # Combine all features into a single list
    feat = ratings + [length_scaled]
    
    return feat

# Ablation study for 'ratings' feature
def feature_without_ratings(datum, max_length):
    # Create one-hot encoded features for beer styles
    feat = [0] * len(categories)
    if datum['beer/style'] in categories:
        feat[catID[datum['beer/style']]] = 1

    # Extract the review length
    length = len(datum['review/text'])

    # Scale the 'length' feature to be between 0 and 1
    length_scaled = length / max_length

    # Combine all features into a single list
    feat += [length_scaled]
    
    return feat

# Ablation study for 'length' feature
def feature_without_length(datum):
    # Create one-hot encoded features for beer styles
    feat = [0] * len(categories)
    if datum['beer/style'] in categories:
        feat[catID[datum['beer/style']]] = 1

    # Extract the review ratings
    ratings = [datum['review/aroma'], datum['review/appearance'], datum['review/palate'], datum['review/taste'], datum['review/overall']]

    # Combine all features into a single list
    feat += ratings
    
    return feat

# Logistic Regression model training with only 'ratings' and 'length' features
logreg_ratings_length = linear_model.LogisticRegression(C=1, class_weight='balanced')

# Logistic Regression model training with only 'style' and 'length' features
logreg_style_length = linear_model.LogisticRegression(C=1, class_weight='balanced')

# Logistic Regression model training with only 'style' and 'ratings' features
logreg_style_ratings = linear_model.LogisticRegression(C=1, class_weight='balanced')

# Training the models after removing each feature
logreg_ratings_length.fit([feature_without_style(d, max_length_train) for d in dataTrain], yTrain)
logreg_style_length.fit([feature_without_ratings(d, max_length_train) for d in dataTrain], yTrain)
logreg_style_ratings.fit([feature_without_length(d) for d in dataTrain], yTrain)

# Making predictions on the test set using the models without each feature
yTestPred_ratings_length = logreg_ratings_length.predict([feature_without_style(d, max_length_train) for d in dataTest])
yTestPred_style_length = logreg_style_length.predict([feature_without_ratings(d, max_length_train) for d in dataTest])
yTestPred_style_ratings = logreg_style_ratings.predict([feature_without_length(d) for d in dataTest])

# Computing the test BER for each ablated feature
test_ber_ratings_length = 1 - balanced_accuracy_score(yTest, yTestPred_ratings_length)
test_ber_style_length = 1 - balanced_accuracy_score(yTest, yTestPred_style_length)
test_ber_style_ratings = 1 - balanced_accuracy_score(yTest, yTestPred_style_ratings)

# Printing the results of the ablation study
print(f'Test BER without Beer Style: {test_ber_ratings_length}')
print(f'Test BER without Ratings: {test_ber_style_length}')
print(f'Test BER without Length: {test_ber_style_ratings}')


# In[5]:


# Task 5: 10 items have the highest Jaccard similarity compared to item ‘B00KCHRKD6’

"""
report both similarities and item IDs for the 10 most similar items.
"""

import gzip
from collections import defaultdict

path = "amazon_reviews_us_Musical_Instruments_v1_00.tsv.gz"
f = gzip.open(path, 'rt', encoding="utf8")

header = f.readline()
header = header.strip().split('\t')
dataset = []

pairsSeen = set()

for line in f:
    fields = line.strip().split('\t')
    d = dict(zip(header, fields))
    ui = (d['customer_id'], d['product_id'])
    if ui in pairsSeen:
        # print("Skipping duplicate user/item:", ui)
        continue
    pairsSeen.add(ui)
    d['star_rating'] = int(d['star_rating'])
    d['helpful_votes'] = int(d['helpful_votes'])
    d['total_votes'] = int(d['total_votes'])
    dataset.append(d)

dataTrain = dataset[:int(len(dataset)*0.9)]
dataTest = dataset[int(len(dataset)*0.9):]

usersPerItem = defaultdict(set) # Maps an item to the users who rated it
itemsPerUser = defaultdict(set) # Maps a user to the items that they rated
itemNames = {}
ratingDict = {} # To retrieve a rating for a specific user/item pair
reviewsPerUser = defaultdict(list)

for d in dataTrain:
    user, item = d['customer_id'], d['product_id']
    usersPerItem[item].add(user)
    itemsPerUser[user].add(item)
    itemNames[item] = d['product_title']
    ratingDict[user, item] = d['star_rating']
    reviewsPerUser[user].append(d)

userAverages = {}
itemAverages = {}

for u in itemsPerUser:
    userRatings = [ratingDict[u, i] for i in itemsPerUser[u]]
    userAverages[u] = sum(userRatings) / len(userRatings)

for i in usersPerItem:
    itemRatings = [ratingDict[u, i] for u in usersPerItem[i]]
    itemAverages[i] = sum(itemRatings) / len(itemRatings)

def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    return numer / denom

def mostSimilar(i, N):
    similarities = []
    for i2 in itemNames:
        if i2 != i:
            sim = Jaccard(usersPerItem[i], usersPerItem[i2])
            similarities.append((sim, i2))
    similarities.sort(reverse=True)
    return similarities[:N]

query = 'B00KCHRKD6'
ms = mostSimilar(query, 10)
for sim, item in ms:
    print(f"Jaccard Similarity: {sim}, Item ID: {item}")


# In[6]:


# Task 6: implement a rating prediction model based on the similarity function

"""
split the data into 90% train and 10% testing portions. 
when computing similarities return the item’s average rating if no similar items exist (i.e., if the denominator is zero).
or the global average rating if that item hasn’t been seen before. 
all averages should be computed on the training set only. 
report the MSE of this rating prediction function on the test set when Sim(i, j) = Jaccard(i, j).
"""

import gzip
from collections import defaultdict

# Load the data
path = "amazon_reviews_us_Musical_Instruments_v1_00.tsv.gz"
f = gzip.open(path, 'rt', encoding="utf8")

header = f.readline()
header = header.strip().split('\t')
dataset = []

pairsSeen = set()

for line in f:
    fields = line.strip().split('\t')
    d = dict(zip(header, fields))
    ui = (d['customer_id'], d['product_id'])
    if ui in pairsSeen:
        # print("Skipping duplicate user/item:", ui)
        continue
    pairsSeen.add(ui)
    d['star_rating'] = int(d['star_rating'])
    d['helpful_votes'] = int(d['helpful_votes'])
    d['total_votes'] = int(d['total_votes'])
    dataset.append(d)

dataTrain = dataset[:int(len(dataset) * 0.9)]
dataTest = dataset[int(len(dataset) * 0.9):]

# Preprocess the data
usersPerItem = defaultdict(set)  # Maps an item to the users who rated it
itemsPerUser = defaultdict(set)  # Maps a user to the items that they rated
reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)
ratingDict = {}  # To retrieve a rating for a specific user/item pair

for d in dataTrain:
    usersPerItem[d['product_id']].add(d['customer_id'])
    itemsPerUser[d['customer_id']].add(d['product_id'])
    reviewsPerUser[d['customer_id']].append(d)
    reviewsPerItem[d['product_id']].append(d)

# Implement the Jaccard similarity function
def Jaccard(s1, s2):
    intersection = len(s1 & s2)
    union = len(s1 | s2)
    return intersection / union if union != 0 else 0

ratingMean = sum([d['star_rating'] for d in dataTrain])/len(dataTrain)

# Calculate user and item averages
usersAverages = {}
itemsAverages = {}

for user in reviewsPerUser.keys():
    usersAverages[user]=sum(d['star_rating'] for d in reviewsPerUser[user])/len(reviewsPerUser[user])

for item in reviewsPerItem.keys():
    itemsAverages[item]=sum(d['star_rating'] for d in reviewsPerItem[item])/len(reviewsPerItem[item])
        
def predictRating_Jaccard(user,item):
    if item not in usersPerItem:
        return ratingMean
    ratings = []
    similarities = []
    for d in reviewsPerUser[user]:
        k = d['product_id']
        if k != item:
            ratings.append(d['star_rating'] - itemsAverages[k])
            similarities.append(Jaccard(usersPerItem[item],usersPerItem[k]))
    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return itemsAverages[item] + sum(weightedRatings) / sum(similarities)
    else:
        # User hasn't rated any similar items
        return itemsAverages[item]

def MSE(predictions, labels):
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)

labels = [d['star_rating'] for d in dataTest]
simPredictions = [predictRating_Jaccard(d['customer_id'],d['product_id']) for d in dataTest]

mse = MSE(simPredictions,labels)
print("Mean Squared Error (MSE) on the test set:", mse)


# In[7]:


# Task 7: time-weight collaborative filtering

"""
design a decay function that outperforms (in terms of the MSE) the trivial function f(tu,j) = 1.
documente any design choices you make.
"""

import gzip
from collections import defaultdict
import math

# Load the data
path = "amazon_reviews_us_Musical_Instruments_v1_00.tsv.gz"
f = gzip.open(path, 'rt', encoding="utf8")

header = f.readline()
header = header.strip().split('\t')
dataset = []

pairsSeen = set()

for line in f:
    fields = line.strip().split('\t')
    d = dict(zip(header, fields))
    ui = (d['customer_id'], d['product_id'])
    if ui in pairsSeen:
        # print("Skipping duplicate user/item:", ui)
        continue
    pairsSeen.add(ui)
    d['star_rating'] = int(d['star_rating'])
    d['helpful_votes'] = int(d['helpful_votes'])
    d['total_votes'] = int(d['total_votes'])
    dataset.append(d)

dataTrain = dataset[:int(len(dataset) * 0.9)]
dataTest = dataset[int(len(dataset) * 0.9):]

# Preprocess the data
usersPerItem = defaultdict(set)  # Maps an item to the users who rated it
itemsPerUser = defaultdict(set)  # Maps a user to the items that they rated
reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)
ratingDict = {}  # To retrieve a rating for a specific user/item pair

for d in dataTrain:
    usersPerItem[d['product_id']].add(d['customer_id'])
    itemsPerUser[d['customer_id']].add(d['product_id'])
    reviewsPerUser[d['customer_id']].append(d)
    reviewsPerItem[d['product_id']].append(d)

# Implement the Jaccard similarity function
def Jaccard(s1, s2):
    intersection = len(s1 & s2)
    union = len(s1 | s2)
    return intersection / union if union != 0 else 0

ratingMean = sum([d['star_rating'] for d in dataTrain]) / len(dataTrain)

# Calculate user and item averages
usersAverages = {}
itemsAverages = {}

for user in reviewsPerUser.keys():
    usersAverages[user] = sum(d['star_rating'] for d in reviewsPerUser[user]) / len(reviewsPerUser[user])

for item in reviewsPerItem.keys():
    itemsAverages[item] = sum(d['star_rating'] for d in reviewsPerItem[item]) / len(reviewsPerItem[item])

import math
from datetime import datetime

def time_decay(t1, t2, alpha=0.1):
    """
    A time-based decay function that decreases the impact of ratings farther in time.
    t1, t2: Datetime objects representing the timestamps of the ratings.
    alpha: A decay parameter that determines the rate of decay.
    """
    time_diff = abs((t1 - t2).days/10000)
    return math.exp(-alpha * time_diff)

def convert_to_datetime(date_string):
    """
    Convert the date string in the format 'YYYY-MM-DD' to a datetime object.
    """
    return datetime.strptime(date_string, '%Y-%m-%d')
    
def predictRating_Jaccard_with_decay(user, item, target_time):
    if item not in usersPerItem:
        return ratingMean
    ratings = []
    similarities = []
    for d in reviewsPerUser[user]:
        k = d['product_id']
        if k != item:
            ratings.append(d['star_rating'] - itemsAverages[k])
            exp_rate = time_decay(convert_to_datetime(d['review_date']),convert_to_datetime(target_time))  # assuming 'review_date' is in Unix timestamp format
            similarities.append(Jaccard(usersPerItem[item], usersPerItem[k]) * exp_rate)
    if sum(similarities) != 0:
        weighted_ratings = [x * y for x, y in zip(ratings, similarities)]
        return itemsAverages[item] + sum(weighted_ratings) / sum(similarities)
    else:
        # User hasn't rated any similar items
        return itemsAverages[item]

def MSE(predictions, labels):
    differences = [(x - y) ** 2 for x, y in zip(predictions, labels)]
    return sum(differences) / len(differences)

labels = [d['star_rating'] for d in dataTest]
time_decayed_predictions = [predictRating_Jaccard_with_decay(d['customer_id'], d['product_id'], d['review_date']) for d in dataTest]

mse_time_decay = MSE(time_decayed_predictions, labels)
print("Mean Squared Error (MSE) on the test set with time decay:", mse_time_decay)


# In[8]:


# Codes below are used for answer.txt generation

def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N

answers = {}


# In[9]:


### Question 1

answers['Q1'] = [0.16130237168160533, 0.1607838024608832]

assertFloatList(answers['Q1'], 2)


# In[10]:


### Question 2

answers['Q2'] = [0.14214313781636712, 0.14301810257164882]

assertFloatList(answers['Q2'], 2)


# In[11]:


### Question 3

answers['Q3'] = [0.1, 0.1432824729179234, 0.14544069028123996]

assertFloatList(answers['Q3'], 3)


# In[12]:


### Question 4

answers['Q4'] = [0.3139492057092712, 0.16109632033831978, 0.14658340274812243]

assertFloatList(answers['Q4'], 3)


# In[13]:


### Question 5

answers['Q5'] = ms

assertFloatList([m[0] for m in ms], 10)


# In[14]:


### Question 6

answers['Q6'] = 1.7165666373341593

assertFloat(answers['Q6'])


# In[15]:


### Question 7

answers['Q7'] = ["Add time decay with decay rate 0.1", 1.716566096661067]

assertFloat(answers['Q7'][1])


# In[16]:


f = open("answers_hw2.txt", 'w')
f.write(str(answers) + '\n')
f.close()


# In[ ]:




