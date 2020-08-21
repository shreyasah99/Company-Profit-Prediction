import pandas as pd
import numpy as np
import pickle

df=pd.read_csv('datafinal.csv')
df.drop(['Unnamed: 0'], axis=1, inplace=True)

X = df.iloc[:,0:4]
y = df.iloc[:,4]

X= pd.DataFrame(X)
X.replace(["California",'Florida','New York'], [1,2,3] , inplace=True)

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.25,random_state=10)

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg=linreg.fit(X_train,y_train)

pickle.dump(linreg, open('model.pkl','wb'))




