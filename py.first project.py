import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# --- 1. Data Collection ---
def fetch_air_quality_data(city, api_key):
    """Fetches air quality data from an API."""
    #The URL was not an API endpoint, it was a link to a Kaggle dataset
    #Kaggle datasets are not returned as JSON
    #This example will use a different API that returns JSON
    url = f"https://api.iqair.com/v1/airquality?city=New%20York&apiKey=4f6a40ad-f170-4ecb-b017-a70e15bd9f3b"
    response = requests.get(url, verify=False)
    if response.status_code == 200:
        data = response.json()
        return data['list'][0]['components']
    else:
        print(f"Error fetching data: {response.status_code}")
        return None

# --- 2. Data Preprocessing ---
def preprocess_data(pollution_data):
    """Cleans and formats the fetched air quality data."""
    if pollution_data is None:
        return None

    # Example: Extract relevant pollutants and create a DataFrame
    data = {
        'co': pollution_data['co'],
        'no': pollution_data['no'],
        'no2': pollution_data['no2'],
        'o3': pollution_data['o3'],
        'so2': pollution_data['so2'],
        'pm2_5': pollution_data['pm2_5'],
        'pm10': pollution_data['pm10'],
        'nh3': pollution_data['nh3'],
    }
    df = pd.DataFrame([data])
    return df

# --- 3. Model Training ---
def train_model(data):
    """Trains a machine learning model to predict air quality."""
    if data is None or data.empty:
        print("No data available for training.")
        return None

    # Example: Using 'co' as the target variable and others as features
    X = data[['no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']]
    y = data['co']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model

# --- 4. Prediction ---
def predict_air_quality(model, new_data):
    """Predicts air quality using the trained model."""
    if model is None:
        print("Model not trained. Cannot predict.")
        return None

    predictions = model.predict(new_data)
    return predictions

# --- Main Execution ---
if __name__ == "__main__":
    # Replace with your actual API key
    api_key = "4f6a40ad-f170-4ecb-b017-a70e15bd9f3b" #This needs to be a valid key from IQAir

    # Fetch data for a specific city
    city_name = "San Francisco"
    pollution_data = fetch_air_quality_data(city_name, api_key)

    # Preprocess the data
    df = preprocess_data(pollution_data)

    # Train the model (if data is available)
    model = train_model(df)

    # Example new data for prediction (replace with actual data)
    new_data = pd.DataFrame({
        'no': [10],
        'no2': [50],
        'o3': [20],
        'so2': [20],
        'pm2_5': [20],
        'pm10': [20],
        'nh3': [20],
    })

    # Make predictions (if model is trained)
    predictions = predict_air_quality(model, new_data)