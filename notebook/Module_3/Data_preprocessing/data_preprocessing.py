#Data Preprocessing Template

#Import Libraries

import numpy as np #for numerical computation
import matplotlib.pyplot as plt #for visualization
import pandas as pd #for data manipulation

#import dataset
dataset = pd.read_csv("Salary_Data.csv")

#To create the matrix of the independent variable
x = dataset.iloc[:,:3].values

#To create the matrix the dependent variable
y = dataset.iloc[:,3:4].values

#Handle missing data

#A.Count missing data
missing_data = dataset.isnull().sum().sort_values(ascending = False)

#B.Impute missing data
from sklearn.impute import SimpleImputer
simple_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
simple_imputer = simple_imputer.fit(x[:, 1:3])
x[:, 1:3] = simple_imputer.transform(x[:, 1:3])

#To Encode the Categorical Data

#A.To Encode the Categorical Data(Country) in Variable X to become ordinal
from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
x[:,0] = labelencoder_x.fit_transform(x[:,0])

#B.To Encode the Categorical Data(Country) in Variable X to become nominal
# using dummy variable
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

column_transformer = ColumnTransformer([("Country", OneHotEncoder
                                         (categories='auto'), 
                                         [0])], remainder = "passthrough")
x = column_transformer.fit_transform(x)
x=x.astype(float)

#C.To Encode the Categorical Data which is purchase in variable y to become 
#numerical
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)

#To Split the whole dataset into training dataset and testing dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x,y, train_size=0.8, 
                                                    test_size=0.2, 
                                                    random_state=0)


#To perform feature scaling

#A. Standardization feature scaling
from sklearn.preprocessing import StandardScaler
standardscaler = StandardScaler()
X_train_standard = X_train.copy()
X_train_standard[:,3:5] = standardscaler.fit_transform(X_train_standard[:,3:5])

X_test_standard = X_test.copy()
X_test_standard[:,3:5] = standardscaler.transform(X_test_standard[:,3:5])

#To perform Normalizatoin Feature Scaling
from sklearn.preprocessing import MinMaxScaler
minmaxscaler = MinMaxScaler()
X_train_normal = X_train.copy()
X_train_normal[:,3:5] = minmaxscaler.fit_transform(X_train_normal[:,3:5])

X_test_normal = X_test.copy()
X_test_normal[:,3:5] = minmaxscaler.transform(X_train_normal[:,3:5])



















