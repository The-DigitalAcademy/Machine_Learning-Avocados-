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

# # Load the dataset
# data = pd.read_csv('avocado.csv')

# # Perform one-hot encoding for categorical variables
# encoder = OneHotEncoder(drop="first")
# encoded_data = pd.get_dummies(data, columns=['Date','Total Volume','4046','4225','4770','type','year','region'])

# # Split the dataset into features (X) and target variable (y)
# features = ['Date','TotalVolume','4046','4225','4770','TotalBags','SmallBags','LargeBags','XLargeBags','type','year','region']
# target = 'AveragePrice'
# X = encoded_data.drop(target, axis=1)
# Y = encoded_data[target]
# scale = StandardScaler()
# scaledX = scale.fit_transform(X)

# X_train, X_test, y_train, y_test = train_test_split(scaledX, Y, test_size=0.2, random_state=42)

# Create and train the model
# regressor = RandomForestRegressor(n_estimators=100, random_state=0)
# regressor.fit(X_train, y_train)


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
        # train_y_pred = model.predict(X_train)

        # mse = mean_squared_error(y_train, train_y_pred)
        # r2 = r2_score(y_train, train_y_pred)
        # st.write(f"Mean Squared Error (Train): {mse}")
        # st.write(f"R-squared (Train): {r2}")

        # mse = mean_squared_error(y_test, test_y_pred)
        # r2 = r2_score(y_test, test_y_pred)
        # st.write(f"Mean Squared Error (Test): {mse}")
        # st.write(f"R-squared (Test): {r2}")

        # # Perform one-hot encoding for user input
        # encoded_user_input = pd.get_dummies(user_input, columns=['season', 'region', 'month', 'type'])
        # encoded_user_input = encoded_user_input.reindex(columns=X.columns, fill_value=0)

        # Make predictions using the trained model
        # price_prediction = regressor.predict(scale.transform(encoded_user_input))

        # Display the predicted price
        st.write("Predicted Average Price:", test_y_pred[0])
        st.markdown("Display Accuracy:")
        # for key, value in results.items():
        #     st.write(f"- {key}: {value}")
        st.markdown("Display AveragePrice:")
        # for key, value in results.items():
        #      st.write(f"- {key}: {value}")

st.balloons()
