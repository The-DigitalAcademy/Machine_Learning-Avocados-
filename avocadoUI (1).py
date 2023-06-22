import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(layout='wide', page_title='Wizards Market', page_icon='🥑')

st.title('Wizards Market')

st.write("~~~~~~~~~")
st.markdown("<h1 style='text-align: center; color: white;'>Wizards Market</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: white;'>🥑</h1>", unsafe_allow_html=True)
st.write("~~~~~~~~~")
st.markdown("<h2 style='text-align: center; color: white;'>Welcome to Wizards Market!</h2>", unsafe_allow_html=True)
st.write("~~~~~~~~~~~")

with st.form(key='display options'):
    st.write("1. Display Avocado Average Price Prediction\n")
    new_tot = st.number_input("Total Number Of Avocado-4046: ")
    new_num = st.number_input("Total Number Of Avocado-4770: ")
    new_avo = st.number_input("Total Number Of Avocado-4225: ")
    tot_bags = st.number_input("Total Bags: ")
    seasons = st.selectbox("Choose A Season:", ["Winter", "Spring", "Summer", "Autumn"])
    region = st.text_input("Enter Your Region: ")
    month_avo = st.selectbox("C Which Month: ")
    type_avo = st.selectbox("Choose A Type:", ["Conventional", "Organic"])
    type_algorthm = st.selectbox("Choose A Machine Learning Algorthm:", ["Decision Tree", "Random forest", 'Lasso', 'Ridge', 'Linear Regression',' Adaboost Regression', 'Stacking Regression'])
    submit_details = st.form_submit_button('Submit!')

    if submit_details:
        # Create a DataFrame with user input
        user_input = pd.DataFrame({
            '4046': [new_tot],
            '4770': [new_num],
            '4225': [new_avo],
            'Total Bags': [tot_bags],
            'season_' + seasons: [1],
            'region_' + region: [1],
            'month_' + month_avo: [1],
            'type_' + type_avo: [1]
        })
        
        if type_algorthm == "Decision Tree":
            model_load_path = "dt.pkl"
            with open(model_load_path,'rb') as file:
                model = pickle.load(file)
                        
        
        test_y_pred = model.predict(user_input)

        st.write("Predicted Average Price:", test_y_pred[0])
        st.markdown("Display Accuracy:")
    
        st.markdown("Display AveragePrice:")
       

st.balloons()
