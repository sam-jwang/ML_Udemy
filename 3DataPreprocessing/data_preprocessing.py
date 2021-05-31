import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer

# Trying to predict if someone will purchase something given:
#   Country, Age, Salary

dataset = pd.read_csv('Data.csv')
# X is features (what we use to predict)
# iloc -> locates indexes of cols we want
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(X)
print(y)

# Takes care of nan, fills with mean
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# Specifying columns w missing values, but in general good to just include all bc you don't know where 
#   missing values are
imputer.fit(X[:, 1:3])
# .transform() returns updated columns
X[:, 1:3] = imputer.transform(X[:, 1:3])

print(X)