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

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test


dtree = DecisionTreeClassifier()
# Run decision tree with all data
dtree = dtree.fit(X, y)

# Train Decision Tree Classifer
#dtree = dtree.fit(X_train, y_train)

# predict response
y_pred = dtree.predict(X_test)


# show results
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

fig = plt.figure(figsize=(100,25))

tree.plot_tree(dtree, feature_names=attributes_minus_class, class_names=['v-good', 'good', 'acc', 'unacc'], filled=True, rounded=True, fontsize=10)
fig.savefig("tree.png")