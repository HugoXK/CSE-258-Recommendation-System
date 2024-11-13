#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import gzip
import math
from collections import defaultdict
import numpy as np
from sklearn import linear_model
import random
import statistics


# In[2]:


def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[3]:


answers = {}


# In[4]:


z = gzip.open("train.json.gz")


# In[5]:


dataset = []
for l in z:
    d = eval(l)
    dataset.append(d)


# In[6]:


z.close()


# In[7]:


### Question 1


# In[8]:


def MSE(y, ypred):
    return np.mean((np.array(y) - np.array(ypred)) ** 2)


# In[9]:


def MAE(y, ypred):
    return np.mean(np.abs(np.array(y) - np.array(ypred)))


# In[10]:


reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)

for d in dataset:
    u,i = d['userID'],d['gameID']
    reviewsPerUser[u].append(d)
    reviewsPerItem[i].append(d)
    
for u in reviewsPerUser:
    reviewsPerUser[u].sort(key=lambda x: x['date'])
    
for i in reviewsPerItem:
    reviewsPerItem[i].sort(key=lambda x: x['date'])


# In[11]:


def feat1(d):
    return [1, d['hours']]  # '1' for the bias term (θ0)  


# In[12]:


# Prepare the data for linear regression
X = [feat1(d) for d in dataset]
y = [len(d['text']) for d in dataset]


# In[13]:


# Initialize the linear regression model
mod = linear_model.LinearRegression()

# Fit the model
mod.fit(X, y)

# Make predictions
predictions = mod.predict(X)


# In[14]:


# Reporting the value of θ1 and the Mean Squared Error
theta_1 = mod.coef_[1]  # The coefficient for 'hours' feature
mse_q1 = MSE(y, predictions)


# In[15]:


answers['Q1'] = [theta_1, mse_q1]


# In[16]:


assertFloatList(answers['Q1'], 2)


# In[17]:


### Question 2


# In[18]:


# Calculate the median hours across the dataset
median_hours = np.median([d['hours'] for d in dataset])


# In[19]:


# Define the new feature extraction function incorporating the given transforms
def feat2(d):
    hours = d['hours']
    hours_transformed = [
        1,  # for θ0
        hours,  # for θ1
        math.log2(hours + 1),  # for θ2
        hours**2,  # for θ3
        1 if hours > median_hours else 0  # for θ4
    ]
    return hours_transformed


# In[20]:


X = [feat2(d) for d in dataset]


# In[21]:


mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X,y)
predictions = mod.predict(X)


# In[22]:


# Calculate the MSE
mse_q2 = MSE(y, predictions)


# In[23]:


answers['Q2'] = mse_q2


# In[24]:


assertFloat(answers['Q2'])


# In[25]:


### Question 3


# In[26]:


# Function to extract features based on the given binary indicators
def feat3(d):
    hours = d['hours']
    return [1,  # Bias term for θ0
            1 if hours > 1 else 0,    # δ(h>1) for θ1
            1 if hours > 5 else 0,    # δ(h>5) for θ2
            1 if hours > 10 else 0,   # δ(h>10) for θ3
            1 if hours > 100 else 0,  # δ(h>100) for θ4
            1 if hours > 1000 else 0] # δ(h>1000) for θ5


# In[27]:


X = [feat3(d) for d in dataset]


# In[28]:


mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X,y)
predictions = mod.predict(X)


# In[29]:


# Calculate the MSE
mse_q3 = MSE(y, predictions)


# In[30]:


answers['Q3'] = mse_q3


# In[31]:


assertFloat(answers['Q3'])


# In[32]:


### Question 4


# In[33]:


# Define a function to extract features from the dataset
def feat4(d):
    return [1, len(d['text'])]  # Adding a constant term for θ0 and the review length for θ1


# In[34]:


X = [feat4(d) for d in dataset]
y = [d['hours'] for d in dataset]  # Extracting the number of hours played as the target variable


# In[35]:


mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X,y)
predictions = mod.predict(X)


# In[36]:


# Calculate MSE and MAE
mse = MSE(y, predictions)
mae = MAE(y, predictions)


# In[37]:


print(f"MSE:{mse},MSE:{mae}")


# In[38]:


answers['Q4'] = [mse, mae, "MSE is sensitive to outliers, while MAE gives equal weight to all errors. The choice depends on the dataset's characteristics."]


# In[39]:


assertFloatList(answers['Q4'][:2], 2)


# In[40]:


### Question 5


# In[41]:


# Prepare the features (X)
X = [feat4(d) for d in dataset]


# In[42]:


# Prepare the labels (y) by applying the transformation log2(hours + 1)
y_trans = [d['hours_transformed'] for d in dataset]


# In[43]:


mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X,y_trans)
predictions_trans = mod.predict(X)


# In[44]:


# Calculate MSE using the transformed target variable
mse_trans = MSE(y_trans, predictions_trans)


# In[45]:


# Calculate the original hours values for MSE comparison
original_hours = [d['hours'] for d in dataset]


# In[46]:


# Transform the predictions back to the original scale by inverting the transformation
predictions_untrans = [2**pred - 1 for pred in predictions_trans]


# In[47]:


# Calculate MSE on the untransformed (original hours) scale
mse_untrans = MSE(original_hours, predictions_untrans)


# In[48]:


answers['Q5'] = [mse_trans, mse_untrans]


# In[49]:


assertFloatList(answers['Q5'], 2)


# In[50]:


### Question 6


# In[51]:


def one_hot(hours, dim=100):
    # Create a one-hot encoded vector for 'hours' with 'dim' dimensions
    # Any value for 'hours' >= dim is encoded in the last position
    vector = [0] * dim
    index = min(int(hours), dim - 1)  # To ensure the index is within the bounds of 'vector'
    vector[index] = 1
    return vector

def feat6(d):
    # Extract the one-hot encoding feature for the hours played
    return one_hot(d['hours'])


# In[52]:


X = [feat6(d) for d in dataset]
y = [len(d['text']) for d in dataset]


# In[53]:


Xtrain, Xvalid, Xtest = X[:len(X)//2], X[len(X)//2:(3*len(X))//4], X[(3*len(X))//4:]
ytrain, yvalid, ytest = y[:len(X)//2], y[len(X)//2:(3*len(X))//4], y[(3*len(X))//4:]


# In[54]:


models = {}
mses = {}
bestC = None

# Loop over the list of regularization strengths
for c in [1, 10, 100, 1000, 10000]:
    # Create and fit the Ridge regression model with current regularization strength
    model = linear_model.Ridge(alpha=c)
    model.fit(Xtrain, ytrain)

    # Store the model
    models[c] = model

    # Predict and calculate MSE on validation set
    yvalid_pred = model.predict(Xvalid)
    mse_valid = MSE(yvalid, yvalid_pred)
    mses[c] = mse_valid


# In[55]:


# Find the best alpha (regularization strength) based on the validation MSE
bestC = min(mses, key=mses.get)


# In[56]:


# Predict and calculate MSE on test set using the best model
predictions_test = models[bestC].predict(Xtest)


# In[57]:


mse_valid = mses[bestC]


# In[58]:


mse_test = MSE(ytest, predictions_test)


# In[59]:


answers['Q6'] = [bestC, mse_valid, mse_test]


# In[60]:


assertFloatList(answers['Q6'], 3)


# In[61]:


### Question 7


# In[62]:


times = [d['hours_transformed'] for d in dataset]
median = statistics.median(times)


# In[63]:


notPlayed = [d for d in dataset if d['hours'] < 1]
nNotPlayed = len(notPlayed)


# In[64]:


answers['Q7'] = [median, nNotPlayed]


# In[65]:


assertFloatList(answers['Q7'], 2)


# In[66]:


### Question 8


# In[67]:


def feat8(d):
    return [len(d['text'])]  # wrap the feature in a list as scikit-learn expects a 2D array    


# In[68]:


X = [feat8(d) for d in dataset]
y = [d['hours_transformed'] > median for d in dataset]


# In[69]:


mod = linear_model.LogisticRegression(class_weight='balanced')
mod.fit(X,y)
predictions = mod.predict(X) # Binary vector of predictions


# In[70]:


# Define the rates function to calculate the confusion matrix components
def rates(predictions, y):
    TP = sum(p and t for p, t in zip(predictions, y))
    TN = sum(not p and not t for p, t in zip(predictions, y))
    FP = sum(p and not t for p, t in zip(predictions, y))
    FN = sum(not p and t for p, t in zip(predictions, y))
    return TP, TN, FP, FN


# In[71]:


TP, TN, FP, FN = rates(predictions, y)


# In[72]:


BER = 0.5 * ((FP / (len(y) - sum(y))) + (FN / sum(y)))  # The average of the false positive rate and false negative rate


# In[73]:


answers['Q8'] = [TP, TN, FP, FN, BER]


# In[74]:


assertFloatList(answers['Q8'], 5)


# In[75]:


### Question 9


# In[76]:


# Get the probabilities of the positive class
probs = mod.predict_proba(X)[:, 1]  # Probabilities of the positive class

# Sort the instances by their probability of being positive
sorted_indices = probs.argsort()[::-1]
sorted_scores = probs[sorted_indices]
sorted_labels = [y[idx] for idx in sorted_indices]


# In[77]:


precs = []

for k in [5, 10, 100, 1000]:
    # Find the k-th score (0-indexed)
    threshold_score = sorted_scores[k-1]
    
    # Find all indices where the score is greater than or equal to the threshold
    # This handles ties by including all instances with the same score as the k-th element
    tie_indices = np.where(sorted_scores >= threshold_score)[0]
    
    # Adjust k to account for ties
    adjusted_k = tie_indices[-1] + 1
    
    # Compute precision
    precision_at_k = np.sum(sorted_labels[:adjusted_k]) / adjusted_k
    precs.append(precision_at_k)


# In[78]:


answers['Q9'] = precs


# In[79]:


assertFloatList(answers['Q9'], 4)


# In[80]:


### Question 10


# In[128]:


y_trans = [d['hours_transformed'] for d in dataset]


# In[86]:


mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X,y_trans)
predictions_trans = mod.predict(X)


# In[87]:


thresholds = np.linspace(0, 1, num=101) # You can use more or fewer points.
best_threshold = None
lowest_BER = float('inf')

for threshold in thresholds:
    # Apply threshold to regression predictions to get binary classification
    predictions_thresh = predictions_trans >= threshold
    # Calculate TP, TN, FP, FN using confusion matrix
    TP, TN, FP, FN = rates(predictions_thresh, y_trans)

    # Compute BER
    BER = 0.5 * (FN / (TP + FN) + FP / (TN + FP))
    
    # Store the best threshold and lowest BER
    if BER < lowest_BER:
        lowest_BER = BER
        best_threshold = threshold


# In[88]:


answers['Q10'] = [best_threshold, lowest_BER]


# In[89]:


answers['Q10']


# In[90]:


assertFloatList(answers['Q10'], 2)


# In[91]:


### Question 11


# In[92]:


dataTrain = dataset[:int(len(dataset)*0.9)]
dataTest = dataset[int(len(dataset)*0.9):]


# In[93]:


userMedian = defaultdict(list)
itemMedian = defaultdict(list)


# In[94]:


for entry in dataTrain:
    userMedian[entry['userID']].append(entry['hours'])
    itemMedian[entry['gameID']].append(entry['hours'])

# Compute medians using the collected playtimes
userMedian = {user: statistics.median(playtimes) for user, playtimes in userMedian.items()}
itemMedian = {item: statistics.median(playtimes) for item, playtimes in itemMedian.items()}


# In[95]:


answers['Q11'] = [itemMedian['g35322304'], userMedian['u55351001']]


# In[96]:


assertFloatList(answers['Q11'], 2)


# In[97]:


### Question 12


# In[98]:


global_playtimes = [d['hours'] for d in dataTrain]
global_median = statistics.median(global_playtimes)


# In[99]:


def f12(u,i):
    # Function returns a single value (0 or 1)
    # Check if item i has been seen before
    if i in itemMedian and itemMedian[i] > global_median:
        # If seen, return 1 if the item's median time played is above the global median
        return 1
    else:
        # If the item hasn't been seen, check if the user's median time is above the global median
        return 1 if i not in itemMedian and userMedian.get(u, 0) > global_median else 0


# In[100]:


preds = [f12(d['userID'], d['gameID']) for d in dataTest]


# In[101]:


y = [1 if d['hours'] > global_median else 0 for d in dataTest]


# In[102]:


accuracy = sum(pred == true for pred, true in zip(preds, y)) / len(y)


# In[103]:


answers['Q12'] = accuracy


# In[104]:


assertFloat(answers['Q12'])


# In[105]:


### Question 13


# In[106]:


usersPerItem = defaultdict(set) # Maps an item to the users who rated it
itemsPerUser = defaultdict(set) # Maps a user to the items that they rated
itemNames = {}

for d in dataset:
    user,item = d['userID'], d['gameID']
    usersPerItem[item].add(user)
    itemsPerUser[user].add(item)


# In[107]:


def Jaccard(s1, s2):
    # Jaccard Similarity: |Intersection(s1, s2)| / |Union(s1, s2)|
    numerator = len(s1.intersection(s2))
    denominator = len(s1.union(s2))
    return numerator / denominator if denominator > 0 else 0  # prevent division by zero


# In[108]:


# Find the Most Similar Items Function
def mostSimilar(i, func, N=10):
    similarities = []
    users_of_item_i = usersPerItem[i]
    for other_item in usersPerItem:
        if other_item == i: continue  # Skip comparing the item to itself
        sim = func(users_of_item_i, usersPerItem[other_item])
        similarities.append((sim, other_item))
    # Sort based on similarity in descending order and return the top N
    similarities.sort(reverse=True)
    return similarities[:N]


# In[109]:


ms = mostSimilar(dataset[0]['gameID'], Jaccard, 10)


# In[110]:


answers['Q13'] = [ms[0][0], ms[-1][0]]


# In[111]:


assertFloatList(answers['Q13'], 2)


# In[112]:


### Question 14


# In[113]:


ratingDict = {}

for d in dataset:
    u,i,playtime = d['userID'], d['gameID'], d['hours']
    lab = 1 if playtime > global_median else -1 # Set the label based on a rule
    ratingDict[(u,i)] = lab


# In[114]:


def Cosine(i1, i2):
    # Between two items
    inter = usersPerItem[i1].intersection(usersPerItem[i2])
    numer = 0
    denom1 = 0
    denom2 = 0
    for u in inter:
        numer += ratingDict[(u,i1)]*ratingDict[(u,i2)]
    for u in usersPerItem[i1]:
        denom1 += ratingDict[(u,i1)]**2
    for u in usersPerItem[i2]:
        denom2 += ratingDict[(u,i2)]**2
    denom = math.sqrt(denom1) * math.sqrt(denom2)
    if denom == 0: return 0
    return numer / denom


# In[115]:


def mostSimilar14(i, func, N=10):
    similarities = []
    users = usersPerItem[i]
    for i2 in usersPerItem:
        if i2 == i: continue
        sim = func(i, i2)
        similarities.append((sim,i2))
    similarities.sort(reverse=True)
    return similarities[:N]


# In[116]:


ms = mostSimilar14(dataset[0]['gameID'], Cosine, 10)


# In[117]:


answers['Q14'] = [ms[0][0], ms[-1][0]]


# In[118]:


assertFloatList(answers['Q14'], 2)


# In[119]:


### Question 15


# In[120]:


ratingDict = {}

for d in dataset:
    u,i = d['userID'], d['gameID']
    lab = d['hours_transformed']# Set the label based on a rule
    ratingDict[(u,i)] = lab


# In[121]:


ms = mostSimilar14(dataset[0]['gameID'], Cosine, 10)


# In[122]:


answers['Q15'] = [ms[0][0], ms[-1][0]]


# In[123]:


assertFloatList(answers['Q15'], 2)


# In[124]:


f = open("answers_midterm.txt", 'w')
f.write(str(answers) + '\n')
f.close()

