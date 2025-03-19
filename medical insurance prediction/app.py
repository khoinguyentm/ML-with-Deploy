# import the necessary libraries
import numpy as np
import pandas as pd
import pickle as pkl
import streamlit as st
import os

# import the trained model
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
with open(model_path, "rb") as f:
    model = pkl.load(f)

st.header('Medical Insurance Premium Prediction')

gender = st.selectbox('Choose gender', ['Male', 'Female'])
smoker = st.selectbox('Are you a smoker?', ['Yes', 'No'])
age = st.slider('Enter age', 5, 80)
bmi = st.slider('Enter BMI', 5, 100)
region = st.selectbox('Choose regiuon', ['southwest', 'southeast', 'northwest', 'northeast'])
children = st.slider('Enter number of children', 0, 5)

if gender == 'Female':
    gender = 0
else:
    gender = 1
    

if smoker == 'Yes':
    smoker = 1
else:
    smoker = 0
    
if region == 'southwest':
    region = 0
elif region == 'southeast':
    region = 1
elif region == 'northwest':
    region = 2
else:
    region = 3
    
input_data = (age, gender, bmi, children, smoker, region)
input_data = np.array(input_data).reshape(1, -1)

prediction = model.predict(input_data)

display_string = 'Insurance Premium will be $' + str(round(prediction[0])) + ' US dollars'

st.markdown(display_string)
