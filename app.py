import pickle
import numpy as np
import pandas as pd
import streamlit as st


# -------------------------------------------------------
# Load trained model
# -------------------------------------------------------

@st.cache_resource
def load_model():
    with open("house_price_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model


model = load_model()


# -------------------------------------------------------
# App title
# -------------------------------------------------------

st.set_page_config(
    page_title="House Value Prediction",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 House Value Prediction App")
st.write(
    "This app predicts house value using a machine learning regression model."
)

st.warning(
    "This model is trained on the California Housing Dataset, so the locations are California-based."
)


# -------------------------------------------------------
# Area location mapping
# -------------------------------------------------------

area_coordinates = {
    "Los Angeles": {"Latitude": 34.05, "Longitude": -118.24},
    "San Francisco": {"Latitude": 37.77, "Longitude": -122.42},
    "San Diego": {"Latitude": 32.72, "Longitude": -117.16},
    "Sacramento": {"Latitude": 38.58, "Longitude": -121.49},
    "San Jose": {"Latitude": 37.34, "Longitude": -121.89},
    "Fresno": {"Latitude": 36.74, "Longitude": -119.78},
    "Oakland": {"Latitude": 37.80, "Longitude": -122.27},
    "Bakersfield": {"Latitude": 35.37, "Longitude": -119.02},
    "Santa Barbara": {"Latitude": 34.42, "Longitude": -119.70},
    "Long Beach": {"Latitude": 33.77, "Longitude": -118.19},
}


# -------------------------------------------------------
# User inputs
# -------------------------------------------------------

st.subheader("Enter House Details")

area = st.selectbox(
    "Select Area",
    list(area_coordinates.keys())
)

median_income = st.number_input(
    "Median Income in Area",
    min_value=0.5,
    max_value=15.0,
    value=4.0,
    step=0.1
)

house_age = st.number_input(
    "House Age",
    min_value=1,
    max_value=100,
    value=25,
    step=1
)

average_rooms = st.number_input(
    "Average Number of Rooms",
    min_value=1.0,
    max_value=20.0,
    value=5.0,
    step=0.1
)

average_bedrooms = st.number_input(
    "Average Number of Bedrooms",
    min_value=0.5,
    max_value=10.0,
    value=1.0,
    step=0.1
)

population = st.number_input(
    "Population in Area",
    min_value=1,
    max_value=50000,
    value=1500,
    step=100
)

average_occupancy = st.number_input(
    "Average Occupancy per Household",
    min_value=1.0,
    max_value=20.0,
    value=3.0,
    step=0.1
)


# -------------------------------------------------------
# Prediction
# -------------------------------------------------------

if st.button("Predict House Value"):
    latitude = area_coordinates[area]["Latitude"]
    longitude = area_coordinates[area]["Longitude"]

    input_data = pd.DataFrame({
        "MedInc": [median_income],
        "HouseAge": [house_age],
        "AveRooms": [average_rooms],
        "AveBedrms": [average_bedrooms],
        "Population": [population],
        "AveOccup": [average_occupancy],
        "Latitude": [latitude],
        "Longitude": [longitude],
    })

    prediction = model.predict(input_data)[0]

    predicted_price = prediction * 100000

    st.success(f"Estimated House Value in {area}: ${predicted_price:,.2f}")

    st.write("### Input Summary")
    st.dataframe(input_data)


# -------------------------------------------------------
# Extra information
# -------------------------------------------------------

st.markdown("---")
st.write("### About this App")
st.write(
    """
    This project uses machine learning regression models to predict house values.
    The final model was trained using features such as income, house age,
    rooms, bedrooms, population, occupancy, latitude, and longitude.
    """
)