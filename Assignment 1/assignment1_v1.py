#!/usr/bin/env python
# coding: utf-8

# In[2]:


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

# Find out which games a user hasn't played yet.
userPlayedGames = defaultdict(set)

# Populate gameCount dictionary and totalPlayed count.
for user, game, _ in readJSON("train.json.gz"):
    gameCount[game] += 1
    totalPlayed += 1
    userPlayedGames[user].add(game)
    
# Sort games by popularity (number of times played).
mostPopular = [(gameCount[x], x) for x in gameCount]
mostPopular.sort()
mostPopular.reverse()

gamePopularity = {}
for g in gameCount.keys():
    idx = mostPopular.index((gameCount[g],g))
    gamePopularity[g] = 1.0 - idx/len(mostPopular)

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

from sklearn.linear_model import LogisticRegression

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
    popularity = gamePopularity[game]
    X.append([max_similarity, popularity])
    y.append(actual)

# Train a logistic regression model.
clf = LogisticRegression()
clf.fit(X, y)

# Predict on the validation set.
y_pred = clf.predict(X)

# Open the test file and prediction file.
predictions = open("predictions_Played.csv", 'w')

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
    popularity = gamePopularity[g]
    
    # Make a prediction using the logistic regression model.
    pred = clf.predict([[max_similarity, popularity]])[0]  # clf is the trained logistic regression model.
    
    # Write the prediction to the output file.
    predictions.write(u + ',' + g + ',' + str(pred) + '\n')

# Close the prediction file.
predictions.close()


# In[2]:


# import gzip
# from collections import defaultdict
# import random

# # Load data with functions from stub
# def readGz(path):
#     for l in gzip.open(path, 'rt'):
#         yield eval(l)

# def readJSON(path):
#     for l in gzip.open(path, 'rt'):
#         d = eval(l)
#         u = d['userID']
#         try:
#             g = d['gameID']
#         except Exception as e:  # In case the 'gameID' attribute is missing.
#             g = None
#         yield u, g, d

# allHours = []
# for l in readJSON("train.json.gz"):
#     allHours.append(l)

# # Split the data into training and validation sets.
# hoursTrain = allHours[:165000]
# hoursValid = allHours[165000:]

# # Data structures for the would-play baseline (from baseline.py)
# gameCount = defaultdict(int)  # Dictionary to keep track of how many times each game was played.
# totalPlayed = 0  # Total number of games played.

# # Find out which games a user hasn't played yet.
# userPlayedGames = defaultdict(set)

# # Populate gameCount dictionary and totalPlayed count.
# for user, game, _ in readJSON("train.json.gz"):
#     gameCount[game] += 1
#     totalPlayed += 1
#     userPlayedGames[user].add(game)
    
# # Sort games by popularity (number of times played).
# mostPopular = [(gameCount[x], x) for x in gameCount]
# mostPopular.sort()
# mostPopular.reverse()

# gamePopularity = {}
# for g in gameCount.keys():
#     idx = mostPopular.index((gameCount[g],g))
#     gamePopularity[g] = 1.0 - idx/len(mostPopular)

# # Create validation set which includes both positive and negative samples.
# allGames = list(gameCount.keys())  # All distinct games in our dataset.
# validationSet = []

# # Generate negative samples for the valid set.
# for user, game, _ in hoursValid:
#     negativeGame = random.choice(allGames)
#     while negativeGame in userPlayedGames[user]:
#         negativeGame = random.choice(allGames)
#     validationSet.append((user, game, 1))        # Positive example: The actual game played by the user
#     validationSet.append((user, negativeGame, 0)) # Negative example: A randomly chosen game (not played by the user)

# # Create a dictionary with games as keys and sets of users who played them as values.
# gameUsers = defaultdict(set)
# for user, game, _ in hoursTrain:
#     gameUsers[game].add(user)

# # Calculate Jaccard similarity between two sets.
# def jaccard_similarity(set1, set2):
#     intersection = len(set1 & set2)
#     union = len(set1 | set2)
#     return intersection / union if union != 0 else 0

# # Precompute Jaccard similarities for all game pairs.
# game_pair_similarities = defaultdict(dict)
# all_games = list(gameUsers.keys())

# for i, game1 in enumerate(all_games):
#     for j, game2 in enumerate(all_games):
#         if j <= i:
#             continue  # No need to compute similarity twice for the same pair
#         similarity = jaccard_similarity(gameUsers[game1], gameUsers[game2])
#         game_pair_similarities[game1][game2] = similarity
#         game_pair_similarities[game2][game1] = similarity

# from sklearn.linear_model import LogisticRegression

# # Prepare the features and labels.
# X = []
# y = []

# for user, game, actual in validationSet:
#     # Feature 1: Jaccard similarity
#     similarity = 0
#     count = 0
#     for game_prime in userPlayedGames[user]:
#         if game_prime in game_pair_similarities[game]:
#             similarity += game_pair_similarities[game][game_prime]
#             count += 1
#     similarity /= count
#     # Feature 2: Popularity
#     popularity = gamePopularity[game]
#     X.append([similarity, popularity])
#     y.append(actual)

# # Train a logistic regression model.
# clf = LogisticRegression()
# clf.fit(X, y)

# # Predict on the validation set.
# y_pred = clf.predict(X)

# # Open the test file and prediction file.
# predictions = open("predictions_Played.csv", 'w')

# for l in open("pairs_Played.csv"):
#     if l.startswith("userID"):
#         # Write the header to the output file.
#         predictions.write(l)
#         continue
#     u, g = l.strip().split(',')
    
#     # Extract features for the given (u, g) pair.
#     # Feature 1: Jaccard similarity.
#     similarity = 0
#     count = 0
#     if u in userPlayedGames:  # Make sure the user exists in the training data.
#         for game_prime in userPlayedGames[u]:
#             if game_prime in game_pair_similarities[game]:
#                 similarity += game_pair_similarities[game][game_prime]
#                 count += 1
#         similarity /= count
#     # Feature 2: Popularity.
#     popularity = gamePopularity[g]
    
#     # Make a prediction using the logistic regression model.
#     pred = clf.predict([[similarity, popularity]])[0]  # clf is the trained logistic regression model.
    
#     # Write the prediction to the output file.
#     predictions.write(u + ',' + g + ',' + str(pred) + '\n')

# # Close the prediction file.
# predictions.close()


# In[10]:


from tqdm.notebook import tqdm
import numpy as np

# Calculate global average
trainHours = [r[2]['hours_transformed'] for r in hoursTrain]
globalAverage = sum(trainHours) / len(trainHours)

# Initialize dictionaries
hoursPerUserItem = defaultdict(float)
gamesPerUser = defaultdict(int)
usersPerItem = defaultdict(int)
betaU = defaultdict(float)
betaI = defaultdict(float)

# Populate the dictionaries
for user, game, d in hoursTrain:
    hour = d['hours_transformed']
    if user not in gamesPerUser:
        gamesPerUser[user] = set()
    if game not in usersPerItem:
        usersPerItem[game] = set()
    hoursPerUserItem[(user,game)] = hour
    gamesPerUser[user].add(game)
    usersPerItem[game].add(user)

alpha = globalAverage  # Initialize alpha

def iterate(lamb):
    newAlpha = 0
    newBetaU = {}
    newBetaI = {}
    
    for key, value in hoursPerUserItem.items():
        user, game = key
        hour = value
        newAlpha += (hour-betaU[user]-betaI[game])
    newAlpha /= len(hoursPerUserItem.items())
    
    for u in betaU.keys():
        newU = 0.0
        for g in gamesPerUser[u]:
            hour = hoursPerUserItem[(u,g)]
            newU += (hour - newAlpha - betaI[g])
        newU /= (lamb + len(gamesPerUser[u]))    
        newBetaU[u] = newU

    for i in betaI.keys():
        newI = 0.0
        for u in usersPerItem[i]:
            hour = hoursPerUserItem[(u,i)]
            newI += (hour - newAlpha - betaU[u])
        newI /= (lamb + len(usersPerItem[i]))    
        newBetaI[i] = newI
            
    return newAlpha, newBetaU, newBetaI

# Initialize variables to store the best lambda and its corresponding MSE
best_lambda = None
best_mse = float('inf')

# Iterate over a range of lambda values
for lamb in np.arange(4,5.6,0.1):  # Adjust this range and values as necessary
    
    print(f'-------------lambda={lamb}-------------')
    # Initialize beta values for this lambda
    for u in gamesPerUser.keys():
        betaU[u] = 0

    for g in usersPerItem.keys():
        betaI[g] = 0

    alpha = globalAverage  # Reset alpha

    # Perform the iterations
    for i in tqdm(range(100)):
        alpha, betaU, betaI = iterate(lamb)
        current_mse = 0.0
        valid_mse = 0.0
        for user, game, d in hoursValid:
            hour = d['hours_transformed']
            prediction = alpha + betaU.get(user,0)+ betaI.get(game,0)
            current_mse += ((hour-prediction)**2)/len(hoursValid)
        if (abs(valid_mse-current_mse)<1e-6):
            break
        valid_mse = current_mse

    # Calculate MSE for this lambda
    
    print(f"Lambda: {lamb}, MSE: {valid_mse:.7f}")

    # Update best lambda and MSE if current MSE is lower
    if valid_mse < best_mse:
        best_mse = valid_mse
        best_lambda = lamb

# Print the best lambda and its MSE
print(f"Best lambda: {best_lambda}, with MSE: {best_mse:.7f}")


# In[19]:


# Perform the iterations
for i in tqdm(range(100)):
    alpha, betaU, betaI = iterate(best_lambda)
    current_mse = 0.0
    valid_mse = 0.0
    for user, game, d in hoursValid:
        hour = d['hours_transformed']
        prediction = alpha + betaU.get(user,0)+ betaI.get(game,0)
        current_mse += ((hour-prediction)**2)/len(hoursValid)
    if (abs(valid_mse-current_mse)<1e-5):
        break
    valid_mse = current_mse
    
# Open the test file and prediction file.        
predictions = open("predictions_Hours.csv", 'w')

for l in open("pairs_Hours.csv"):
    if l.startswith("userID"):
        # Write the header to the output file.
        predictions.write(l)
        continue
    u, g = l.strip().split(',')
        
    bu, bi = betaU[u], betaI[g]
    # Write the prediction to the output file.
    predictions.write(u + ',' + g + ',' + str(alpha+bu+bi) + '\n')

# Close the prediction file.
predictions.close()

