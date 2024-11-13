#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Task 1: construct negative samples for the validation set

"""
sample a negative entry by randomly choosing a game that user hasn’t played for each entry (user,game) in the validation set.
evaluate the performance (accuracy) of the baseline model on the validation set you have built.
"""

import gzip
from collections import defaultdict
import random

# Load data with functions from stub
def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)

def readJSON(path):
    for l in gzip.open(path, 'rt'):
        d = eval(l)
        u = d['userID']
        try:
            g = d['gameID']
        except Exception as e:  # In case the 'gameID' attribute is missing.
            g = None
        yield u, g, d

allHours = []
for l in readJSON("train.json.gz"):
    allHours.append(l)

# Split the data into training and validation sets.
hoursTrain = allHours[:165000]
hoursValid = allHours[165000:]

# Data structures for the would-play baseline (from baseline.py)
gameCount = defaultdict(int)  # Dictionary to keep track of how many times each game was played.
totalPlayed = 0  # Total number of games played.

# Populate gameCount dictionary and totalPlayed count.
for user, game, _ in readJSON("train.json.gz"):
    gameCount[game] += 1
    totalPlayed += 1

# Sort games by popularity (number of times played).
mostPopular = [(gameCount[x], x) for x in gameCount]
mostPopular.sort()
mostPopular.reverse()

# Find the top half of the most popular games to create a return set.
return1 = set()  # Set of games that are most popular.
count = 0
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    # Cover at least half of the total plays.
    if count > totalPlayed/2:
        break

# Find out which games a user hasn't played yet.
userPlayedGames = defaultdict(set)
for user, game, _ in readJSON("train.json.gz"):
    userPlayedGames[user].add(game)

# Create validation set which includes both positive and negative samples.
allGames = list(gameCount.keys())  # All distinct games in our dataset.
validationSet = []

# Generate negative samples for the valid set.
for user, game, _ in hoursValid:
    negativeGame = random.choice(allGames)
    while negativeGame in userPlayedGames[user]:
        negativeGame = random.choice(allGames)
    validationSet.append((user, game, 1))        # Positive example: The actual game played by the user
    validationSet.append((user, negativeGame, 0)) # Negative example: A randomly chosen game (not played by the user)

# Evaluate the accuracy of the baseline model on this validation set.
correctPredictions = 0

for user, game, actual in validationSet:
    prediction = 1 if game in return1 else 0
    if prediction == actual:
        correctPredictions += 1

# Calculate accuracy.
accuracy = correctPredictions / len(validationSet)
print(f"Accuracy of the baseline model on the validation set: {accuracy:.4f}")


# In[2]:


# Task 2: improve the model performance with a better threshold

"""
find a better threshold and report its performance on your validation set.
"""

def get_most_popular_games(threshold, mostPopular, totalPlayed):
    """Return a set of games considered 'popular' for a given threshold."""
    popular_games = set()
    count = 0
    for ic, i in mostPopular:
        count += ic
        popular_games.add(i)
        if count > totalPlayed * threshold:
            break
    return popular_games

def evaluate_popularity_threshold(threshold, mostPopular, totalPlayed, validationSet):
    """Evaluate and return the accuracy of a given popularity threshold on the validation set."""
    popular_games = get_most_popular_games(threshold, mostPopular, totalPlayed)
    correct_predictions = sum(1 for user, game, actual in validationSet if (game in popular_games) == bool(actual))
    return correct_predictions / len(validationSet)

# Search for the best threshold by iterating over a range of possible values.
best_accuracy = 0
best_threshold = 0
for threshold in [i*0.01 for i in range(101)]:  # Iterating from 0 to 1 with a step of 0.01
    accuracy = evaluate_popularity_threshold(threshold, mostPopular, totalPlayed, validationSet)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = threshold

print(f"Best threshold: {best_threshold:.2f}")
print(f"Accuracy on validation set using best threshold: {best_accuracy:.4f}")


# In[3]:


# Task 3: Jaccard similarity-based threshold

"""
given a pair (u, g) in the validation set, consider all training items.
compute the Jaccard similarity.
predict as ‘played’ if the maximum of these Jaccard similarities exceeds a threshold. 
report the performance on validation set.
"""

# Create a dictionary with games as keys and sets of users who played them as values.
gameUsers = defaultdict(set)
for user, game, _ in hoursTrain:
    gameUsers[game].add(user)

# Calculate Jaccard similarity between two sets.
def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0

# Precompute Jaccard similarities for all game pairs.
game_pair_similarities = defaultdict(dict)
all_games = list(gameUsers.keys())

for i, game1 in enumerate(all_games):
    for j, game2 in enumerate(all_games):
        if j <= i:
            continue  # No need to compute similarity twice for the same pair
        similarity = jaccard_similarity(gameUsers[game1], gameUsers[game2])
        game_pair_similarities[game1][game2] = similarity
        game_pair_similarities[game2][game1] = similarity

def predict_played_optimized(u, g, threshold):
    max_similarity = 0
    for game_prime in userPlayedGames[u]:
        if game_prime in game_pair_similarities[g]:
            similarity = game_pair_similarities[g][game_prime]
            max_similarity = max(max_similarity, similarity)
    return 1 if max_similarity > threshold else 0

# Now, search for the best threshold using the optimized function.
best_threshold = 0
best_accuracy = 0

for threshold in [i*0.01 for i in range(101)]:
    correct_predictions = sum(1 for user, game, actual in validationSet if predict_played_optimized(user, game, threshold) == actual)
    accuracy = correct_predictions / len(validationSet)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = threshold

print(f"Best threshold for Jaccard similarity: {best_threshold:.2f}")
print(f"Accuracy on validation set using best threshold: {best_accuracy:.4f}")


# In[4]:


# Task 4: incorporate Jaccard-based and popularity based threshold

"""
incorporate both a Jaccard-based threshold and a popularity based threshold. 
report the performance on your validation set
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Prepare the features and labels.
X = []
y = []

for user, game, actual in validationSet:
    # Feature 1: Jaccard similarity
    max_similarity = 0
    for game_prime in userPlayedGames[user]:
        if game_prime in game_pair_similarities[game]:
            similarity = game_pair_similarities[game][game_prime]
            max_similarity = max(max_similarity, similarity)
    # Feature 2: Popularity
    is_popular = int(game in get_most_popular_games(best_threshold, mostPopular, totalPlayed))
    X.append([max_similarity, is_popular])
    y.append(actual)

# Split X and y into training and test datasets (optional, but typically a good idea).
# Note: You might want to have a separate test set to evaluate this.

# Train a logistic regression model.
clf = LogisticRegression()
clf.fit(X, y)

# Predict on the validation set.
y_pred = clf.predict(X)
accuracy = accuracy_score(y, y_pred)

print(f"Accuracy of the combined model on the validation set: {accuracy:.4f}")


# In[5]:


# Task 5:

"""
use the files ‘pairs Played.txt’ to find the reviewerID/itemID pairs about which we have to make predictions. 
use that data, run the above model and upload your solution to the Assignment 1 gradescope.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Open the test file and prediction file.
predictions = open("HWpredictions_Played.csv", 'w')

for l in open("pairs_Played.csv"):
    if l.startswith("userID"):
        # Write the header to the output file.
        predictions.write(l)
        continue
    u, g = l.strip().split(',')
    
    # Extract features for the given (u, g) pair.
    # Feature 1: Jaccard similarity.
    max_similarity = 0
    if u in userPlayedGames:  # Make sure the user exists in the training data.
        for game_prime in userPlayedGames[u]:
            if game_prime in game_pair_similarities[game]:
                similarity = game_pair_similarities[game][game_prime]
                max_similarity = max(max_similarity, similarity)
    
    # Feature 2: Popularity.
    is_popular = int(game in get_most_popular_games(best_threshold, mostPopular, totalPlayed))
    
    # Make a prediction using the logistic regression model.
    pred = clf.predict([[max_similarity, is_popular]])[0]  # clf is the trained logistic regression model.
    
    # Write the prediction to the output file.
    predictions.write(u + ',' + g + ',' + str(pred) + '\n')

# Close the prediction file.
predictions.close()


# In[41]:


# Task 6: time played predictor

"""
fit a predictor by fitting the mean and the two bias terms.
use a regularization parameter of λ = 1. 
report the MSE on the validation set.

"""

# Calculate global average
trainHours = [r[2]['hours_transformed'] for r in hoursTrain]
globalAverage = sum(trainHours) / len(trainHours)

# Initialize dictionaries
hoursPerUser = defaultdict(float)
hoursPerItem = defaultdict(float)
gamesPerUser = defaultdict(int)
usersPerItem = defaultdict(int)
betaU = defaultdict(float)
betaI = defaultdict(float)

# Populate the dictionaries
for user, game, d in hoursTrain:
    hoursPerUser[user] += d['hours_transformed']
    hoursPerItem[game] += d['hours_transformed']
    gamesPerUser[user] += 1
    usersPerItem[game] += 1

alpha = globalAverage  # Initialize alpha

# Regularization parameter
lamb = 1

def iterate(lamb):
    newAlpha = 0
    for user, game, d in hoursTrain:
        newAlpha += d['hours_transformed'] - (betaU[user] + betaI[game])
    alpha = newAlpha / len(hoursTrain)

    for user in betaU:
        newBetaU = 0
        for r in hoursTrain:
            if r[0] == user:
                _, game, d = r
                newBetaU += d['hours_transformed'] - (alpha + betaI[game])
        betaU[user] = newBetaU / (lamb + gamesPerUser[user])

    for game in betaI:
        newBetaI = 0
        for r in hoursTrain:
            if r[1] == game:
                user, _, d = r
                newBetaI += d['hours_transformed'] - (alpha + betaU[user])
        betaI[game] = newBetaI / (lamb + usersPerItem[game])

from tqdm.notebook import tqdm # for visulization

# Perform iterations to refine alpha, betaU, and betaI
for _ in tqdm(range(10)):  # Use tqdm here for progress bar
    iterate(lamb)

# Calculate MSE on the validation set
mse_sum = 0
for user, game, d in hoursValid:
    prediction = alpha + betaU[user] + betaI[game]
    mse_sum += (d['hours_transformed'] - prediction) ** 2

mse = mse_sum / len(hoursValid)
print(f"MSE on validation set: {mse:.4f}")


# In[23]:


# Task 7: report the user and game IDs

"""
report the user and game IDs that have the largest and smallest values of β.
"""

# Find the user with the largest and smallest beta values
max_betaU_user = max(betaU, key=betaU.get)
min_betaU_user = min(betaU, key=betaU.get)

# Find the game with the largest and smallest beta values
max_betaI_game = max(betaI, key=betaI.get)
min_betaI_game = min(betaI, key=betaI.get)

print(f"User with the largest beta value: {max_betaU_user} with value {betaU[max_betaU_user]:.4f}")
print(f"User with the smallest beta value: {min_betaU_user} with value {betaU[min_betaU_user]:.4f}")
print(f"Game with the largest beta value: {max_betaI_game} with value {betaI[max_betaI_game]:.4f}")
print(f"Game with the smallest beta value: {min_betaI_game} with value {betaI[min_betaI_game]:.4f}")


# In[39]:


# Task 8: find a better value of λ

"""
find a better value of λ using your validation set. 
report the value you chose, its MSE.
"""

def mse(validationSet, alpha, betaU, betaI):
    """Calculate the mean squared error on the validation set."""
    errors = [(r[2]['hours_transformed'] - (alpha + betaU[r[0]] + betaI[r[1]])) ** 2 for r in validationSet]
    return sum(errors) / len(errors)

# Initialize variables to store the best lambda and its corresponding MSE
best_lambda = None
best_mse = float('inf')

# Iterate over a range of lambda values
for lamb in [0.001, 0.01, 0.1, 1, 10, 100]:  # Adjust this range and values as necessary
    
    print(f'-------------lambda={lamb}-------------')
    # Initialize beta values for this lambda
    for u in hoursPerUser:
        betaU[u] = 0

    for g in hoursPerItem:
        betaI[g] = 0

    alpha = globalAverage  # Reset alpha

    # Perform the iterations
    for _ in tqdm(range(10)):  # use 10 here for efficiency
        iterate(lamb)

    # Calculate MSE for this lambda
    current_mse = mse(hoursValid, alpha, betaU, betaI)
    print(f"Lambda: {lamb}, MSE: {current_mse:.4f}")

    # Update best lambda and MSE if current MSE is lower
    if current_mse < best_mse:
        best_mse = current_mse
        best_lambda = lamb

# Print the best lambda and its MSE
print(f"Best lambda: {best_lambda}, with MSE: {best_mse:.4f}")


# In[43]:


# Complimentary Experiment for other lambdas as 1 is the best in previous setting
def mse(validationSet, alpha, betaU, betaI):
    """Calculate the mean squared error on the validation set."""
    errors = [(r[2]['hours_transformed'] - (alpha + betaU[r[0]] + betaI[r[1]])) ** 2 for r in validationSet]
    return sum(errors) / len(errors)

# Initialize variables to store the best lambda and its corresponding MSE
best_lambda = None
best_mse = float('inf')

# Iterate over a range of lambda values
for lamb in [0.25, 0.5, 1.5, 2, 2.5, 3, 3.5, 4]:  # Adjust this range and values as necessary
    
    print(f'-------------lambda={lamb}-------------')
    # Initialize beta values for this lambda
    for u in hoursPerUser:
        betaU[u] = 0

    for g in hoursPerItem:
        betaI[g] = 0

    alpha = globalAverage  # Reset alpha

    # Perform the iterations
    for _ in tqdm(range(10)):  # use 10 here for efficiency
        iterate(lamb)

    # Calculate MSE for this lambda
    current_mse = mse(hoursValid, alpha, betaU, betaI)
    print(f"Lambda: {lamb}, MSE: {current_mse:.4f}")

    # Update best lambda and MSE if current MSE is lower
    if current_mse < best_mse:
        best_mse = current_mse
        best_lambda = lamb

# Print the best lambda and its MSE
print(f"Best lambda: {best_lambda}, with MSE: {best_mse:.4f}")


# In[44]:


def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N
    
answers = {}


# In[45]:


answers['Q1'] = 0.6842
assertFloat(answers['Q1'])


# In[46]:


answers['Q2'] = [0.68,0.7056]
assertFloatList(answers['Q2'], 2)


# In[47]:


answers['Q3'] = 0.6688
assertFloat(answers['Q3'])


# In[48]:


answers['Q4'] = 0.6692
assertFloat(answers['Q4'])


# In[49]:


answers['Q5'] = "I confirm that I have uploaded an assignment submission to gradescope"


# In[50]:


answers['Q6'] = 3.0120
assertFloat(answers['Q6'])


# In[51]:


answers['Q7'] = [5.8266, -3.0066, 5.3609, -2.9435]
assertFloatList(answers['Q7'], 4)


# In[52]:


answers['Q8'] = (1.5, 1.0115)
assertFloatList(answers['Q8'], 2)


# In[53]:


f = open("answers_hw3.txt", 'w')
f.write(str(answers) + '\n')
f.close()




