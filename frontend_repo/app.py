import streamlit as st
import requests


param1 = st.slider('Select a number', 1, 10, 3)

param2 = st.slider('Select another number', 1, 10, 3)

url = 'http://localhost:8080/predict'

params = {
    'feature1': param1,  # 0 for Sunday, 1 for Monday, ...
    'feature2': param2
}
response = requests.get(url, params=params)

st.text(response.json())
