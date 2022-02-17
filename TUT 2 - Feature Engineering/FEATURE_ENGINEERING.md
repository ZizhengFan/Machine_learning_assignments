# Variabe Deletion

# import packages
import numpy as np 
import pandas as pd 

# read dataset 
data = pd.read_csv('path/to/data')

#set threshold
threshold = 0.7

# dropping columns with missing value rate higher than threshold
data = data[data.columns[data.isnull().mean() < threshold]]

# MEAN/MEDIAN IMPUTATION

# filling missing values with medians of the columns
data = data.fillna(data.median())

#OR USE VALUE_cOUNTS() METHOD
# filling missing values with medians of the columns
data['column_name'].fillna(data['column_name'].value_counts().idxmax(). inplace=True)

#MIN-MAX SCALER
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# 4 samples/observations and 2 variables/features
data = np.array([[4, 6], [11, 34], [10, 17], [1, 5]])

# create scaler method
scaler = MinMaxScaler(feature_range=(0,1))

# fit and transform the data
scaled_data = scaler.fit_transform(data)

print(scaled_data)

#STANDARD SCALER
from sklearn.preprocessing import StandardScaler
import numpy as np

# 4 samples/observations and 2 variables/features
data = np.array([[4, 1], [11, 1], [10, 4], [1, 11]])

# create scaler method 
scaler = StandardScaler()

# fit and transform the data
scaled_data = scaler.fit_transform(data)

print(scaled_data)

# LABEL ENCODING
# import packages
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder

# intialise data of lists.
data = {'Gender':['male', 'female', 'female', 'male','male'],
        'Country':['Tanzania','Kenya', 'Tanzania', 'Tanzania','Kenya']}
  
# Create DataFrame
data = pd.DataFrame(data)


# create label encoder object
le = LabelEncoder()
  
data['Gender']= le.fit_transform(data['Gender'])
data['Country']= le.fit_transform(data['Country'])

print(data) 

#ONE-HOT ENCODING
# import packages 
import numpy as np 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


# define example
data = np.array(['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot'])

# integer encode
label_encoder = LabelEncoder()

# fit and transform the data
integer_encoded = label_encoder.fit_transform(data)
print(integer_encoded)

# one-hot encode

onehot_encoder = OneHotEncoder(sparse=False)

# reshape the data
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

# fit and transform the data
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

print(onehot_encoded)

# CHI SQUARE TEST
# Load packages
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
  
# Load iris data
iris_dataset = load_iris()
  
# Create features and target
X = iris_dataset.data
y = iris_dataset.target
  
# Convert to categorical data by converting data to integers
X = X.astype(int)
  
# Two features with highest chi-squared statistics are selected
chi2_features = SelectKBest(chi2, k = 2)
X_kbest_features = chi2_features.fit_transform(X, y)
  
# Reduced features
print('Original feature number:', X.shape[1])
print('Reduced feature number:', X_kbest_features.shape[1])

#chi-squared test helps us to select two important independent features out of the original 4 that have the strongest relationship with the target feature


