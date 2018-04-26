# ------------------------------
# IMPORT LIBRARIES
# ------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
# from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ------------------------------
# READ IN DATA
# ------------------------------

games = pd.read_csv("games.csv")

# print(games.columns)
# print(games.shape) # x = games, y = data points describing each game

# ------------------------------
# PLOTTING TARGET VARIABLES - AVERAGE RATING
# ------------------------------

# # Make a histogram of all the ratings in the average_rating column.
# plt.hist(games["average_rating"])

# # Show the plot.
# plt.show()

# # Print the first row of all the games with zero scores.
# # The .iloc method on dataframes allows us to index by position.
# print(games[games["average_rating"] == 0].iloc[0])
# # Print the first row of all the games with scores greater than 0.
# print(games[games["average_rating"] > 0].iloc[0])

# ------------------------------
# REMOVE THE 0 RATINGS
# ------------------------------

# Remove any rows without user reviews.
games = games[games["users_rated"] > 0]
# Remove any rows with missing values.
games = games.dropna(axis=0)

# ------------------------------
# USE KMEANS TO CLUSTER GAMES
# ------------------------------

# Initialize the model with 2 parameters -- number of clusters and random state.
kmeans = KMeans(n_clusters=4, random_state=1)
# Get only the numeric columns from games.
good_columns = games._get_numeric_data()
# Fit the model using the good columns.
kmeans.fit(good_columns)
# Get the cluster assignments.
labels = kmeans.labels_

# ------------------------------
# USING PCA TO PLOT A 2-DIMENSIONAL GRAPH
# ------------------------------

# Create a PCA model.
pca_2 = PCA(2)
# print(good_columns)
# Fit the PCA model on the numeric columns from earlier.
plot_columns = pca_2.fit_transform(good_columns)
# print(plot_columns)
# print(plot_columns[:,0])
# print(plot_columns[:,1])
# Make a scatter plot of each game, shaded according to cluster assignment.
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=labels)

# Show the plot.
plt.show()

# ------------------------------
# FIGURING OUT WHAT TO PREDICT
# ------------------------------

# # Correlation between average_rating and each of the other columns.
# # This will show us which other columns might predict average_rating the best.
# print(games.corr()["average_rating"])

# Remove ratings that might be related to average ratings.
# Get all the columns from the dataframe.
columns = games.columns.tolist()
# Filter the columns to remove ones we don't want.
columns = [c for c in columns if c not in ["bayes_average_rating", "average_rating", "type", "name"]]

# Store the variable we'll be predicting on.
target = "average_rating"
# print(columns)

# ------------------------------
# SPLITTING INTO TRAIN AND TEST SETS
# ------------------------------

# Import a convenience function to split the sets.
# Generate the training set.  Set random_state to be able to replicate results.
# train = games.sample(frac=0.8, random_state=1)
# Select anything not in the training set and put it in the testing set.
# test = games.loc[~games.index.isin(train.index)]
# Print the shapes of both sets.
# print(train.shape)
# print(test.shape)

# ------------------------------
# FIT A LINEAR REGRESSION
# ------------------------------

# Import the linear regression model.
# Initialize the model class.
# model = LinearRegression()
# Fit the model to the training data.
# model.fit(train[columns], train[target])

# ------------------------------
# PREDICTING ERROR
# ------------------------------

# Import the scikit-learn function to compute error.
# Generate our predictions for the test set.
# predictions = model.predict(test[columns])

# Compute error between our test predictions and the actual values.
# print(mean_squared_error(predictions, test[target]))
