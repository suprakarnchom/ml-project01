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

# find k to model
k_neighbors =np.arange(1, 9)
# empty array to store train and test score
train_score = np.empty(len(k_neighbors))
test_score = np.empty(len(k_neighbors))

for i,k in enumerate(k_neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    # วัดประสิทธิภาพ
    train_score[i] = knn.score(x_train, y_train)
    test_score[i] = knn.score(x_test, y_test)

    print(test_score[i]*100)

# plt.title('Compare KNN in Model')
# plt.plot(k_neighbors, test_score, label='Test Score')
# plt.plot(k_neighbors, train_score, label='Train Score')
# plt.legend()
# plt.xlabel('Number of K Neighbors')
# plt.ylabel('Score')
# plt.show()




