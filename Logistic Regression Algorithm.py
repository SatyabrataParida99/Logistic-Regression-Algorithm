import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv(r"D:\FSDS Material\Dataset\Classification\Logistic.csv")

x = data.iloc[:, [2,3]].values 
y = data.iloc[:, -1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state=0)

# Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train, y_train)

# Predicting the test set results

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) 
print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred) 
print(ac)

# This is to get the Classification Report
from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print(cr)

bias = classifier.score(x_train,y_train)
print(bias)

variance = classifier.score(x_test,y_test)
print(variance)

################# Future Prediction #############

data1 = pd.read_csv(r"D:\FSDS Material\Dataset\Classification\Final2.csv")

d2 = data1.copy()

# Extract the relevant columns from the new dataset
data1 = data1.iloc[:, [3, 4]].values


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
M = sc.fit_transform(data1)

y_pred1 = pd.DataFrame()


d2 ['y_pred1'] = classifier.predict(M)
print(d2)

d2.to_csv('pred_model.csv')

# To get the path 
import os
os.getcwd()