import numpy as np
import pandas as pd
import streamlit as st

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# -------------------------------------------------------
# Page setup
# -------------------------------------------------------

st.set_page_config(
    page_title="House Value Prediction",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 House Value Prediction App")
st.write("Predict house values in different California areas using Machine Learning.")


# -------------------------------------------------------
# Train model inside the app
# -------------------------------------------------------

@st.cache_resource
def train_model():
    housing = fetch_california_housing(as_frame=True)

    data = housing.frame
    data.rename(columns={"MedHouseVal": "Price"}, inplace=True)

    X = data.drop("Price", axis=1)
    y = data["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    return model, r2, mae, rmse


model, r2, mae, rmse = train_model()


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
# Sidebar model info
# -------------------------------------------------------

st.sidebar.header("Model Performance")
st.sidebar.write(f"R² Score: {r2:.4f}")
st.sidebar.write(f"MAE: {mae:.4f}")
st.sidebar.write(f"RMSE: {rmse:.4f}")

st.sidebar.info(
    "The model is trained automatically when the app starts. "
    "No large .pkl file is needed."
)


# -------------------------------------------------------
# User input form
# -------------------------------------------------------

st.subheader("Enter House / Area Details")

area = st.selectbox(
    "Select Area",
    list(area_coordinates.keys())
)

median_income = st.slider(
    "Median Income in Area",
    min_value=0.5,
    max_value=15.0,
    value=4.0,
    step=0.1
)

house_age = st.slider(
    "House Age",
    min_value=1,
    max_value=52,
    value=25,
    step=1
)

average_rooms = st.slider(
    "Average Number of Rooms",
    min_value=1.0,
    max_value=15.0,
    value=5.0,
    step=0.1
)

average_bedrooms = st.slider(
    "Average Number of Bedrooms",
    min_value=0.5,
    max_value=5.0,
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

average_occupancy = st.slider(
    "Average Occupancy per Household",
    min_value=1.0,
    max_value=10.0,
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

    st.write("### Input Data Used")
    st.dataframe(input_data)


# -------------------------------------------------------
# Explanation section
# -------------------------------------------------------

st.markdown("---")

st.write("### About this Project")

st.write(
    """
    This app uses a machine learning regression model to predict house values.
    The model is trained on the California Housing Dataset.

    Features used:
    - Median income
    - House age
    - Average rooms
    - Average bedrooms
    - Population
    - Average occupancy
    - Latitude
    - Longitude
    """
)

st.warning(
    "Note: This app predicts prices for California-based areas only because "
    "the training dataset is the California Housing Dataset."
)
)
