import streamlit as st
import joblib
import os

model_path = os.path.join(os.path.dirname(__file__), "iris_model.pkl")
model = joblib.load(model_path)

st.title("Iris Flower Prediction")

sl = st.number_input("Sepal Length (cm)", value=5.1)
sw = st.number_input("Sepal Width (cm)", value=3.5)
pl = st.number_input("Petal Length (cm)", value=1.4)
pw = st.number_input("Petal Width (cm)", value=0.2)

if st.button("Predict"):
    pred = model.predict([[sl, sw, pl, pw]])
    flowers = ["Setosa", "Versicolor", "Virginica"]
    st.success(f"Predicted Iris class: {flowers[pred[0]]}")
