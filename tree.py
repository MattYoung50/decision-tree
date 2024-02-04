import sys
import matplotlib
matplotlib.use('Agg')
#import graphviz

import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


attributes = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df = pandas.read_csv('car.data', names=attributes)
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

X = df[attributes]
Y = df['maint']

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, Y)

fig = plt.figure(figsize=(25,20))

tree.plot_tree(dtree, feature_names=attributes, class_names=['vhigh', 'high', 'med', 'low'], filled=True, rounded=True, fontsize=10)
fig.savefig("tree.png")
# plt.savefig(sys.stdout.buffer)
# sys.stdout.flush()
#plt.show()
#print(df)