import sys
import matplotlib
matplotlib.use('Agg')
#import graphviz

import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

# read data
attributes = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df = pandas.read_csv('car.data', names=attributes)
df.head()

# normalize data into integers
buying_transform = {'vhigh': 1, 'high': 2, 'med': 3, 'low': 4}
maint_transform = {'vhigh': 1, 'high': 2, 'med': 3, 'low': 4}
doors_transform = {'2': 1, '3': 2, '4': 3, '5more': 4}
persons_transform = {'2': 1, '4': 2, 'more': 3}
lug_boot_transform = {'small': 1, 'med': 2, 'big': 3}
safety_transform = {'low': 1, 'med': 2, 'high': 3}
class_transform = {'unacc': 1, 'acc': 2, 'good': 3, 'vgood': 4}
df['buying'] = df['buying'].map(buying_transform)
df['maint'] = df['maint'].map(maint_transform)
df['doors'] = df['doors'].map(doors_transform)
df['persons'] = df['persons'].map(persons_transform)
df['lug_boot'] = df['lug_boot'].map(lug_boot_transform)
df['safety'] = df['safety'].map(safety_transform)
df['class'] = df['class'].map(class_transform)

# Select features and target
attributes_minus_class = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
X = df[attributes_minus_class]
y = df['class']

# In the first step we will split the data in training and remaining dataset
X_train, X_rem, y_train, y_rem = train_test_split(X,y, train_size=0.8, random_state=1)

# Now since we want the valid and test size to be equal (10% each of overall data). 
# we have to define valid_size=0.5 (that is 50% of remaining data)
X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5)
print("X_train: ", X_train.shape), print("y_train: ", y_train.shape)
print("X_valid: ", X_valid.shape), print("y_valid: ", y_valid.shape)
print("X_test: ", X_test.shape), print("y_test: ", y_test.shape)

dtree = DecisionTreeClassifier()
#dtree = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5)
#dtree = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)

# Run decision tree with all data
#dtree = dtree.fit(X, y)
#Train Decision Tree Classifer
dtree = dtree.fit(X_train, y_train)

# predict response
y_pred_test = dtree.predict(X_test)
y_pred_valid = dtree.predict(X_valid)


# show results
print("Accuracy of Test:", metrics.accuracy_score(y_test, y_pred_test))
print("Accuracy of Validation:", metrics.accuracy_score(y_valid, y_pred_valid))

# plot big tree
fig = plt.figure(figsize=(100,25))
# plot small tree
#fig = plt.figure(figsize=(10,10))

tree.plot_tree(dtree, feature_names=attributes_minus_class, class_names=['unacc', 'acc', 'good', 'v-good'], filled=True, rounded=True, fontsize=10)
fig.savefig("tree.png")