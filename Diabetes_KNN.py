from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('diabetes.csv')
# data
x = df.drop('Outcome', axis=1).values
# outcome data
y = df['Outcome'].values
# print(x)

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)

knn = KNeighborsClassifier(n_neighbors=8)

# train
knn.fit(x_train, y_train)

# Predict
y_pred = knn.predict(x_test)

# print(classification_report(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))

print(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))









