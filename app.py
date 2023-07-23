import streamlit as st
import pandas as pd
import numpy as np
import pickle

from src.pipeline.predict_pipeline import PredictPipeline
#import model here.

st.write("""
# Sales price prediction App.

This app predicts the price of the item based on.

Data obtained from [Kaggle bigmart sales data](https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data)

""") 

st.sidebar.header('User Input Features')
st.sidebar.markdown("""
[Example CSV input file]()
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        Item_Weight = st.sidebar.slider('Item Weight', 4, 30, 15)
        Item_Fat_Content = st.sidebar.selectbox('Item Fat content', ('Low Fat', 'Regular'))
        Item_Visibility = st.sidebar.slider('Item visiblity', 0.00, 0.20, 0.10)
        Item_Type = st.sidebar.selectbox('Item Type',('Dairy',
                                                      'Soft Drinks', 
                                                      'Meat',
                                                      'Fruits and Vegetables', 
                                                      'Household', 
                                                      'Baking Goods', 
                                                      'Snack Foods',
                                                      'Frozen Foods',
                                                      'Breakfast',
                                                      'Health and Hygiene',
                                                      'Hard Drinks',
                                                      'Canned',
                                                      'Breads',
                                                      'Starchy Foods',
                                                      'Seafood',
                                                      'Others'))
        Item_MRP = st.sidebar.slider('Item MRP', 30, 300, 100)
        Outlet_Identifier = st.sidebar.selectbox('Outlet identifier', ('OUT049', 'OUT018', 'OUT010', 'OUT013', 'OUT027', 'OUT045', 'OUT017', 'OUT046', 'OUT035', 'OUT019'))
        Outlet_Establishment_Year = st.sidebar.slider('Establishment Year', 1985, 2010, 2000)
        Outlet_Size = st.sidebar.selectbox('Outlet size', ('Small', 'Medium', 'High'))
        Outlet_Location_Type = st.sidebar.selectbox('Outlet location', ('Tier 1', 'Tier 2', 'Tier 3'))
        Outlet_Type = st.sidebar.selectbox('Outlet type', ('Supermarket Type1', 'Supermarket Type2', 'Grocery Store', 'Supermarket Type3'))

        data = {
                'Item_Weight': Item_Weight,
                'Item_Fat_Content': Item_Fat_Content,
                'Item_Visibility' : Item_Visibility,
                'Item_Type' : Item_Type,
                'Item_MRP' : Item_MRP,
                'Outlet_Identifier' : Outlet_Identifier,
                'Outlet_Establishment_Year' : Outlet_Establishment_Year,
                'Outlet_Size' : Outlet_Size,
                'Outlet_Location_Type' : Outlet_Location_Type,
                'Outlet_Type': Outlet_Size
                }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

st.subheader('User Input features')

if uploaded_file is not None:
    st.write(input_df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(input_df)

predict_pipeline = PredictPipeline()

predicted_value= predict_pipeline.predict(input_df)

st.subheader('Prediction')
st.write(predicted_value)
