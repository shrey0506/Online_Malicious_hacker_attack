#importing the necessary library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_data = pd.read_csv('Train.csv')
test_data =  pd.read_csv('Test.csv')

train_data.info()
test_data.info()

train_data.head()
test_data.head()

#creating copy of the dataset
train_copy = train_data.copy()
test_copy = test_data.copy()

train_copy.info()
train_copy.head()

#homogenizing the dataset
train_copy['X_12']=train_copy['X_12'].fillna(0).astype('int')
test_copy['X_12']=test_copy['X_12'].fillna(0).astype('int')

from scipy.stats import mode
train_copy['X_12'].fillna(mode(train_copy['X_12']).mode[0], inplace=True)
test_copy['X_12'].fillna(mode(test_copy['X_12']).mode[0], inplace=True)


#checking if still our daataset contains null value
print(train_copy.isnull().values.sum())
print(test_copy.isnull().values.sum())

print(train_copy['INCIDENT_ID'].value_counts().sum())
print(test_copy['INCIDENT_ID'].value_counts().sum())

from sklearn.model_selection import train_test_split
#making the categorical variable as category to make processing easy
train_copy['INCIDENT_ID']=train_copy['INCIDENT_ID'].astype('category')
train_copy.info()
test_copy['INCIDENT_ID']=test_copy['INCIDENT_ID'].astype('category')
test_copy.info()

#Label Encoding
from sklearn.preprocessing import LabelEncoder
lnc = LabelEncoder()
train_copy['INCIDENT_ID'] = lnc.fit_transform(train_copy['INCIDENT_ID'])
train_copy['INCIDENT_ID'].head()
train_copy=train_copy.drop(['DATE'], axis=1)
train_copy.info()

X = train_copy.iloc[:, :-1].values
y = train_copy.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
#similarly for test dataset
test_copy['INCIDENT_ID'] = lnc.fit_transform(test_copy['INCIDENT_ID'])
test_copy['INCIDENT_ID'].head()
test_copy=test_copy.drop(['DATE'], axis=1)

#using Decesion Tree algorithm
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

test_copy.info()
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

Y_pred = regressor.predict(test_copy)
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(test_copy)


print(y_pred)

a=test_data['INCIDENT_ID']

dict = {'INCIDENT_ID': a, 'MULTIPLE_OFFENSE': Y_pred}  
    
df = pd.DataFrame(dict) 
df.head()
df.shape
type(df)
f = df.to_csv('solfile.csv',columns=['INCIDENT_ID','MULTIPLE_OFFENSE'],index=False)
dataset=pd.read_csv('solfile.csv',index_col=0)
dataset.info


t_m = train_data['MULTIPLE_OFFENSE'].iloc[:15903]
from sklearn.metrics import confusion_matrix 
result = confusion_matrix(t_m,Y_pred)
print(result)



from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
recall=accuracy_score(t_m, Y_pred)
print(recall)

clssfi = classification_report(t_m, Y_pred)
print(clssfi)