# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn import tree
from sklearn.metrics import accuracy_score

# Importing the dataset
dataset = pd.read_csv('data.csv') #Importing the CSV 

dataset.Age.fillna(round(dataset.Age.mean(),0),inplace=True)
dataset.Salary.fillna(round(dataset.Salary.mean(),0),inplace=True)

# Get list of categorical variables
preprocess = make_column_transformer(
    (OneHotEncoder(),['Country']),
    (StandardScaler(),['Age', 'Salary'])
)
pre=preprocess.fit_transform(dataset)

datasetWithEncoder=pd.DataFrame(data=pre)

X = datasetWithEncoder.iloc[:].values 
y = dataset.iloc[:, 3].values 

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection  import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

clf = tree.DecisionTreeClassifier()

clf = clf.fit(X_train, y_train)

y_predict =clf.predict(X_test)

accuracy=accuracy_score(y_test, y_predict)

print(accuracy)
