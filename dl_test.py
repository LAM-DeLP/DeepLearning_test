import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
df_iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df_iris['target'] = iris.target
data_train, data_test, target_train, target_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)
clf = MLPClassifier(hidden_layer_sizes=10, activation='relu',solver='adam', max_iter=1000)
clf.fit(data_train, target_train)
print(clf.score(data_train, target_train))
print("機械学習:"+str(clf.predict(data_test)))
print("せいかい:"+str(target_test))