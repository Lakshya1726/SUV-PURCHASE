import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

cv=pd.read_csv('suv.csv')

x=cv.iloc[:,[2,3]]
y=cv.iloc[:,4]

X_train,X_test,y_train,y_test= train_test_split(x,y,test_size=0.25,random_state=1)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

def app():

    image_path = 'mcr.jpg'
    with st.container():
        st.image(image_path,width=350)
    st.title("SUV CAR Purchase")

    age= st.slider("Select Age", min_value=18, max_value=100, step=1, value=30)
    salary= st.slider("Select Salary",  min_value=10000, max_value=200000, step=1000, value=50000)

    X_new=[[age, salary]]
    X_new_scaled = sc.transform(X_new)
    y_new = model.predict(X_new_scaled)


    if y_new == 1:
        st.write("This person has brought the car")
    else:
        st.write("This person has not brought the car")

if __name__ == "__main__":
    app()