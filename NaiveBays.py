from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Load dataset
iris = load_iris()
x = iris['data']
y = iris['target']

# split data
x_train, x_test, y_train, y_test = train_test_split(x, y)

# Model
model = GaussianNB()

# Train
model.fit(x_train, y_train)

# Predict
y_pred = model.predict(x_test)

# accuracy score
print("Accuracy: ", accuracy_score(y_test, y_pred)*100)
