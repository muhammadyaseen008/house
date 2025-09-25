# app.py
import streamlit as st
import pickle
import pandas as pd

# Load the trained model
with open("AIModel_For_House.pkl", "rb") as f:
    model = pickle.load(f)

# Load the fitted scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.set_page_config(page_title="House Price Prediction", page_icon="🏠")
st.title("🏠 California House Price Prediction App")
st.write("Predict house prices based on California housing dataset features.")

# Sidebar inputs
st.sidebar.header("Input Features")

def user_input_features():
    MedInc = st.sidebar.number_input("Median Income (MedInc)", min_value=0.0, value=3.0)
    HouseAge = st.sidebar.number_input("House Age (HouseAge)", min_value=0.0, value=30.0)
    AveRooms = st.sidebar.number_input("Average Rooms (AveRooms)", min_value=0.0, value=5.0)
    AveBedrms = st.sidebar.number_input("Average Bedrooms (AveBedrms)", min_value=0.0, value=1.0)
    Population = st.sidebar.number_input("Population", min_value=0.0, value=1000.0)
    AveOccup = st.sidebar.number_input("Average Occupancy (AveOccup)", min_value=0.0, value=3.0)
    Latitude = st.sidebar.number_input("Latitude", min_value=32.0, max_value=42.0, value=34.0)
    
    data = {
        'MedInc': MedInc,
        'HouseAge': HouseAge,
        'AveRooms': AveRooms,
        'AveBedrms': AveBedrms,
        'Population': Population,
        'AveOccup': AveOccup,
        'Latitude': Latitude
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Only predict when button is clicked
if st.button("Predict House Price"):
    # Use the trained scaler to transform input features
    input_scaled = scaler.transform(input_df)
    
    # Prediction
    prediction = model.predict(input_scaled)
    
    st.subheader("Prediction")
    st.write(f"Predicted Median House Value: **${prediction[0]*100000:.2f}**")

st.subheader("Input Features")
st.write(input_df)
