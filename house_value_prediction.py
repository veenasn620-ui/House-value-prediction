# House Value Prediction Using Machine Learning Regression Models

import os
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# -------------------------------------------------------
# 1. Load Dataset
# -------------------------------------------------------

def load_dataset(csv_path=None):
    """
    If you have your own dataset, keep it as house_data.csv.
    It must contain a target column like:
    Price / price / SalePrice / house_value / median_house_value

    If no CSV is found, California Housing dataset will be used.
    """

    if csv_path and os.path.exists(csv_path):
        print("Loading dataset from CSV...")
        data = pd.read_csv(csv_path)
        return data

    print("No CSV found. Loading California Housing dataset...")
    housing = fetch_california_housing(as_frame=True)

    data = housing.frame
    data.rename(columns={"MedHouseVal": "Price"}, inplace=True)

    return data


data = load_dataset("house_data.csv")


# -------------------------------------------------------
# 2. Data Understanding
# -------------------------------------------------------

print("\nFirst 5 rows:")
print(data.head())

print("\nDataset shape:")
print(data.shape)

print("\nColumn names:")
print(data.columns.tolist())

print("\nData types:")
print(data.dtypes)

print("\nStatistical summary:")
print(data.describe())

print("\nMissing values:")
print(data.isnull().sum())

print("\nDuplicate rows:")
print(data.duplicated().sum())


# -------------------------------------------------------
# 3. Find Target Column
# -------------------------------------------------------

possible_targets = [
    "Price",
    "price",
    "SalePrice",
    "saleprice",
    "HousePrice",
    "house_price",
    "house_value",
    "median_house_value"
]

target_column = None

for col in possible_targets:
    if col in data.columns:
        target_column = col
        break

if target_column is None:
    raise ValueError(
        "No target price column found. Please rename your price column to 'Price' or 'SalePrice'."
    )

print(f"\nTarget column selected: {target_column}")


# -------------------------------------------------------
# 4. Basic EDA
# -------------------------------------------------------

os.makedirs("plots", exist_ok=True)

plt.figure(figsize=(8, 5))
sns.histplot(data[target_column], kde=True)
plt.title("House Price Distribution")
plt.xlabel("House Price")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("plots/price_distribution.png")
plt.close()

plt.figure(figsize=(8, 5))
sns.boxplot(x=data[target_column])
plt.title("House Price Boxplot")
plt.tight_layout()
plt.savefig("plots/price_boxplot.png")
plt.close()

numeric_data = data.select_dtypes(include=["int64", "float64"])

plt.figure(figsize=(12, 8))
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png")
plt.close()

print("\nEDA plots saved inside 'plots' folder.")


# -------------------------------------------------------
# 5. Data Preprocessing
# -------------------------------------------------------

data = data.drop_duplicates()

X = data.drop(columns=[target_column])
y = data[target_column]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object", "category"]).columns

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])


# -------------------------------------------------------
# 6. Split Dataset
# -------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -------------------------------------------------------
# 7. Train Multiple Models
# -------------------------------------------------------

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    )
}

results = []

trained_models = {}

for model_name, model in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    trained_models[model_name] = pipeline


# -------------------------------------------------------
# 8. Model Comparison
# -------------------------------------------------------

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="R2 Score", ascending=False)

print("\nModel Comparison:")
print(results_df)

results_df.to_csv("model_comparison.csv", index=False)

best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

print(f"\nBest Model: {best_model_name}")


# -------------------------------------------------------
# 9. Hyperparameter Tuning for Random Forest
# -------------------------------------------------------

print("\nRunning hyperparameter tuning for Random Forest...")

rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42))
])

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}

grid_search = GridSearchCV(
    rf_pipeline,
    param_grid,
    cv=3,
    scoring="r2",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

tuned_model = grid_search.best_estimator_
tuned_predictions = tuned_model.predict(X_test)

tuned_mae = mean_absolute_error(y_test, tuned_predictions)
tuned_rmse = np.sqrt(mean_squared_error(y_test, tuned_predictions))
tuned_r2 = r2_score(y_test, tuned_predictions)

print("\nTuned Random Forest Results:")
print("Best Parameters:", grid_search.best_params_)
print("MAE:", tuned_mae)
print("RMSE:", tuned_rmse)
print("R2 Score:", tuned_r2)


# -------------------------------------------------------
# 10. Select Final Model
# -------------------------------------------------------

if tuned_r2 > results_df.iloc[0]["R2 Score"]:
    final_model = tuned_model
    final_model_name = "Tuned Random Forest"
else:
    final_model = best_model
    final_model_name = best_model_name

print(f"\nFinal Selected Model: {final_model_name}")


# -------------------------------------------------------
# 11. Save Model
# -------------------------------------------------------

with open("house_price_model.pkl", "wb") as file:
    pickle.dump(final_model, file)

print("\nModel saved as house_price_model.pkl")


# -------------------------------------------------------
# 12. Prediction Function
# -------------------------------------------------------

def predict_house_price(input_dict):
    """
    Example input:
    {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.984,
        "AveBedrms": 1.023,
        "Population": 322.0,
        "AveOccup": 2.555,
        "Latitude": 37.88,
        "Longitude": -122.23
    }
    """

    input_df = pd.DataFrame([input_dict])
    prediction = final_model.predict(input_df)
    return prediction[0]


# -------------------------------------------------------
# 13. Example Prediction
# -------------------------------------------------------

sample_input = X_test.iloc[0].to_dict()
sample_prediction = predict_house_price(sample_input)

print("\nSample Input:")
print(sample_input)

print("\nPredicted House Value:")
print(sample_prediction)

print("\nActual House Value:")
print(y_test.iloc[0])