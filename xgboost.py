# ------------------------------
# IMPORT LIBRARIES
# ------------------------------
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ------------------------------
# READ IN DATA
# ------------------------------

websites = pd.read_csv("malicious.csv")
# print(websites.columns)
# print(websites.shape) # x = websites, y = data points describing each website

# ------------------------------
# PLOTTING TARGET VARIABLES - AVERAGE RATING
# ------------------------------

# # Make a histogram of all the special websites in the special characters column.
# plt.hist(websites["NUMBER_SPECIAL_CHARACTERS"])
# plt.hist(websites["URL_LENGTH"])
# plt.hist(websites["APP_PACKETS"])

# # Show the plot.
# plt.show()

# # Print the first row of all the websites with zero packets.
# # The .iloc method on dataframes allows us to index by position.
# print(websites[websites["APP_PACKETS"] == 0].iloc[0])
# # Print the first row of all the websites with packets greater than 0.
# print(websites[websites["APP_PACKETS"] > 0].iloc[0])

# # ------------------------------
# # REMOVE THE 0 RATINGS
# # ------------------------------
print("Gradient Boost")
# # Remove any rows without packets.
websites = websites[websites["APP_PACKETS"] > 0]

# # Remove any rows with missing values.
websites = websites.dropna(axis=0)

# # ------------------------------
# # USE XGBOOST TO CLASSIFY WEBSITES
# # ------------------------------

# Initialize the model with 2 parameters -- number of clusters and random state.
xgboost = GradientBoostingClassifier()
# Get only the numeric columns from websites.

good_columns = websites._get_numeric_data()
good_columns = good_columns.drop(["Type"], axis=1)
# Fit the model using the good columns.
xgboost.fit(good_columns, websites["Type"])
# Get the cluster assignments.
print("This is the score: " + str(xgboost.score(good_columns, websites["Type"])))


# # ------------------------------
# # USING PCA TO PLOT A 2-DIMENSIONAL GRAPH
# # ------------------------------

# # Create a PCA model.
# pca_2 = PCA(2)
# # print(good_columns)
# # Fit the PCA model on the numeric columns from earlier.
# plot_columns = pca_2.fit_transform(good_columns)
# # print(plot_columns)
# # print(plot_columns[:,0])
# # print(plot_columns[:,1])
# # Make a scatter plot of each game, shaded according to cluster assignment.
# plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=labels)

# # Show the plot.
# plt.show()

# ------------------------------
# FIGURING OUT WHAT TO PREDICT
# ------------------------------

# # Correlation between app_packets and each of the other columns.
# # This will show us which other columns might predict app_packets the best.
# print(websites.corr()["APP_PACKETS"])

# Remove ratings that might be related to average ratings.
# Get all the columns from the dataframe.
columns = websites.columns.tolist()
# Filter the columns to remove ones we don't want.
columns = [c for c in columns if c in ["SOURCE_APP_PACKETS", "REMOTE_APP_PACKETS", "APP_PACKETS", "SOURCE_APP_BYTES", "REMOTE_APP_BYTES", "APP_BYTES"]]

# Store the variable we'll be predicting on.
target = "APP_PACKETS"

# ------------------------------
# SPLITTING INTO TRAIN AND TEST SETS
# ------------------------------

# Generate the training set.  Set random_state to be able to replicate results.
train = websites.sample(frac=0.8, random_state=1)
# Select anything not in the training set and put it in the testing set.
test = websites.loc[~websites.index.isin(train.index)]
# Print the shapes of both sets.
# print(train.shape)
# print(test.shape)

# ------------------------------
# FIT A LINEAR REGRESSION
# ------------------------------
print("Linear Regression")
# Initialize the model class.
model = LinearRegression()
# Fit the model to the training data.
model.fit(train[columns], train[target])

# ------------------------------
# PREDICTING ERROR
# ------------------------------

# Import the scikit-learn function to compute error.
# Generate our predictions for the test set.
predictions = model.predict(test[columns])

# Compute error between our test predictions and the actual values.
print("This is the mean squared error: " + str(mean_squared_error(predictions, test[target])))
