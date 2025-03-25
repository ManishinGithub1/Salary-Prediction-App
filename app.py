import streamlit as st
import numpy as np
import pickle

model=pickle.load(open(r"C:\Users\ymani\Full Stack Data Science\Salary_Prediction_App\linear_model.pkl",'rb'))

st.title('Salary Prediction App')

st.write("This App predicts the Salary based on the years of experience using Simple Linear Regression")

years_experience=st.number_input("Select the year of experience:",min_value=0.0,max_value=50.0,value=1.0,step=0.5)

if st.button("Predicted Salary"):
    experience_input=np.array([[years_experience]])
    prediction=model.predict(experience_input)
    st.success(f"The Predicted Salary for {years_experience} years of experience is:${prediction[0]:,.2f}")

st.write("The Model was Trained based on the Salary_dataset.csv and build by Y. Manish Kumar")    