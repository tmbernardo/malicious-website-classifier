import pandas as pd
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_json("1500.json")
# ma = np.array(data.as_matrix())
# y_target = ma[:,0]
# X = ma[:,[1,10]]
X = data[['facet_1', 'facet_10', 'facet_2', 'facet_3', 'facet_4', 'facet_5', 'facet_6', 'facet_7', 'facet_8', 'facet_9']]
y_target = data['color']
# print(data.describe())
y_train = set(y_target)


print("# of unique rows: %d" % X.drop_duplicates().shape[0])

gnb = GaussianNB()

y_pred = gnb.fit(X, y_target).predict(X)
y_prob = gnb.predict_proba(X)

print("Percentage of mislabeled points out of a total %d points : %d" % ((X.shape[0],(y_target != y_pred).sum()/X.shape[0]*100)) + "%")

# print((y_target != y_pred).sum())
# print(y_pred)
# print(y_prob)

# max_probabilities = [max(entry) for entry in y_prob]
# print(max_probabilities)

good = np.sum(max(entry) > 0.5 for entry in y_prob)
bad = y_prob.shape[0] - good

print(y_train)
print("Number of good predictions:", good)
print("Number of bad predictions:", bad)

pca_2 = PCA(2)

plot_columns = pca_2.fit_transform(X)

# Make a scatter plot of each game, shaded according to cluster assignment.
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1])
# plt.scatter(max_probabilities, max_probabilities)

plt.show()

