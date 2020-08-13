import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.title('Company Profit Predictor App')
img = Image.open("C:\\Users\\shrey\\Downloads\\profit_photo.jpg")
st.image(img ,width=450 )

df=pd.read_csv('datafinal.csv')
df.drop(['Unnamed: 0'], axis=1, inplace=True)

df['R&D Spend']       = df['R&D Spend'].astype(int)
df['Administration']  = df['Administration'].astype(int)
df['Marketing Spend'] = df['Marketing Spend'].astype(int)
df['Profit'] = df['Profit'].astype(int)

if st.checkbox('Show dataset'):
    st.write(df)

a =  st.sidebar.slider("Amount Spent on R&D?",int(df['R&D Spend'].min()),int(df['R&D Spend'].max()),int(df['R&D Spend'].mean()))
b =  st.sidebar.slider("Amount Spent on Administration?",int(df['Administration'] .min()),int(df['Administration'] .max()),int(df['Administration'] .mean()))
c =  st.sidebar.slider("Amount Spent on Marketing Spend ?",int(df['Marketing Spend'] .min()),int(df['Marketing Spend'] .max()),int(df['Marketing Spend'] .mean()))
d =  st.sidebar.selectbox("Enter state" , ('New York' , 'California' , 'Florida'))


X = df.iloc[:,0:4]
y = df.iloc[:,4]

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.25,random_state=10)


model = LinearRegression()
model=model.fit(X_train,y_train)
predicted=model.predict(X_test)
accuracy=model.score(X_train, y_train)

if d=='New York':
    pred = model.predict([[0,0,1,a, b, c]])
elif d=='California':
    pred = model.predict([[1,0,0,a, b, c]])
else :
    pred = model.predict([[0,1,0,a, b, c]])

if st.sidebar.button('RUN ME!'):
       st.write('Your Profit is',int(pred))
       st.write("Accuracy is ",accuracy)
st.bar_chart(df['State'] )
