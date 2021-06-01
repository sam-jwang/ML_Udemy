import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Trying to predict if someone will purchase something given:
#   Country, Age, Salary

dataset = pd.read_csv('Data.csv')
# X is features (what we use to predict)
# iloc -> locates indexes of cols we want
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Takes care of nan, fills with mean
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# Specifying columns w missing values, but in general good to just include all bc you don't know where 
#   missing values are
imputer.fit(X[:, 1:3])
# .transform() returns updated columns
X[:, 1:3] = imputer.transform(X[:, 1:3])

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

print(X_train)
print(X_test)
print(y_train)
print(y_test)
