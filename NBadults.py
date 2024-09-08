import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def clean_data(dataset):
    for column in dataset.columns:
        if dataset[column].dtype == 'object':
            le = LabelEncoder()
            dataset[column] = le.fit_transform(dataset[column])

    return dataset

def split_feature_class(dataset, feature):
    features = dataset.drop(feature, axis=1) # เอาข้อมูลยกเว้น  income
    labels = dataset[feature].copy() # income data
    return features, labels

dataset = pd.read_csv('adult.csv')
dataset = clean_data(dataset)


# แบ่งข้อมูลเป็น 2 ส่วน คือ train และ test
training_set, test_set = train_test_split(dataset, test_size=0.2)

# แบ่งข้อมูลเป็น 2 ส่วน คือ feature และ class
train_features, train_labels = split_feature_class(training_set, 'income')

# test
test_features, test_labels = split_feature_class(test_set, 'income')


# สร้างโมเดล
model = GaussianNB()
model.fit(train_features, train_labels)

# predict
clf_predict = model.predict(test_features)

print("Accuracy: ", accuracy_score(test_labels, clf_predict)*100)

