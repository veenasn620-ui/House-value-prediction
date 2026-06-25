# House Value Prediction Using Machine Learning

A machine learning web application that predicts house values based on area and property-related features. The app is built using Python, Scikit-learn, and Streamlit.

## Project Overview

This project predicts house values using regression models trained on the California Housing Dataset. Users can enter details such as median income, house age, number of rooms, population, occupancy, and area. The app then predicts the estimated house value.

The final version is deployed as a Streamlit web app so users can access it through a browser and make predictions easily.

## Features

* Predicts house values using machine learning
* User-friendly Streamlit web interface
* Area-based prediction using latitude and longitude
* Model trains automatically inside the app
* No large model file upload required
* Displays model performance metrics
* Easy to run locally or deploy online

## Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Streamlit

## Machine Learning Model

The app uses a regression model trained on the California Housing Dataset.

The input features used are:

* Median income
* House age
* Average number of rooms
* Average number of bedrooms
* Population
* Average occupancy
* Latitude
* Longitude

The target variable is:

* House value

## Project Structure

```text
house-value-prediction/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run the Project Locally

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/house-value-prediction.git
```

### 2. Go to the Project Folder

```bash
cd house-value-prediction
```

### 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

After running the command, the app will open in your browser.

## Requirements

The required Python libraries are listed in `requirements.txt`:

```text
streamlit
numpy
pandas
scikit-learn
```

## How the App Works

1. The app loads the California Housing Dataset.
2. The dataset is split into training and testing data.
3. A regression model is trained using Scikit-learn.
4. The user enters house and area details in the web app.
5. The trained model predicts the estimated house value.
6. The result is displayed on the screen.

## Model Performance

The app displays model performance metrics such as:

* R² Score
* Mean Absolute Error
* Root Mean Squared Error

These metrics help evaluate how well the model predicts house values.

## Future Improvements

Possible improvements for this project include:

* Use a real-world housing dataset with more detailed features
* Add more cities and locations
* Add interactive charts and maps
* Improve model accuracy using hyperparameter tuning
* Deploy the app publicly using Streamlit Community Cloud
* Add user authentication
* Build an advanced dashboard

## Author

Veena Sadasivan Nair

## License

This project is open-source and available for educational purposes.
