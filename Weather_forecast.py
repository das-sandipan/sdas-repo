import requests # Lib to fetch data from API
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # to split data intontraining and test data set
from sklearn.preprocessing import LabelEncoder # to convert categorical value to numerical value
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor # for clasiification and regression
from sklearn.metrics import mean_squared_error # to measure accuracy of prediction model
from datetime import datetime, timedelta # for using date and time

# API KEY Fetching
API_KEY = '' # put actual API Key
BASE_URL = ''   # Base URL for making API Request

# Fetch Current Weather Data
def get_current_weather(city):
    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric" # Construction of API URL
    response = requests.get(url) # send the get request to API
    data = response.jason()
    return {
        'city': data['name'],
        'current_temp': round(data['main'] ['temp']),
        'feels_like': round(data['main'] ['feels_like']),
        'temp_min': round(data['main'] ['temp_min']),
        'temp_max': round(data['main'] ['temp_max']),
        'humidity': round(data['main'] ['humidity']),
        'description': data['weather'][0]['description'],
        'country': data['sys']['country'], # 'sys' is given as how API throws data for country code response
    }
# Read Historical Data 
def get_historical_data(filename):
    df = pd.read_csv(filename) #load CSV file into dataframe
    df = df.dropna() # to remove "NA" value 
    df = df.drop_duplicates() # to remove duplicates
    return df

# Prepare Data for Training
def prepare_data (data):
    le = LabelEncoder(data) # create label encoder instance
    data['WindGustDir'] = le.fit_transform(data['WindGustDir']) # transforming categorical data to numerical : WindGustDir is an attribut column in file
    data['RainTomorrow'] = le.fit_transform(data['RainTomorrow']) # transforming categorical data to numerical : RainTomorrow is an attribut column in file
    # defining feature and target variables (attribute columns from the CSV file)
    x = data[['MinTemp','MaxTemp','WindGustDir','WindGustSpeed','Humidity','Pressure','Temp']] # Features variables
    y = data['RainTomorrow'] # Target Variable

    return x , y , le # Return Feature Variables and Target Variable alongwith '*encoded label*'

# Train Rain Prediction Model
def train_rain_model(x, y):
    x_train , x_test , y_train , y_test = train_test_split (x , y , test_size=0.2 , random_state=42)
    model = RandomForestClassifier(n_estimators=100 , random_state=42)
    model.fit(x_train , y_train) # Train the model

    y_pred = model.predict(x_test) # to make prediction on test set
    print("Mean Square Error for Rain Model")
    print(mean_squared_error(y_test , y_pred))
    return model

# Prep of Regression Data
def prepare_regression_data(data, feature):
    x , y = [] , [] # intialization of feature and target values
    for i in range(len(data) -1):
        x.append(data[feature].iloc[i])
        y.append(data[feature].iloc[i+1])
    x  = np.array(x).reshape(-1 , 1)
    y = np.array(y)
    return x , y

# Train Regression Model
def train_regression_model(x , y):
    model = RandomForestRegressor(n_estimators=100 , random_state=42)
    model.fit(x , y)
    return model

# Predict the Future Value
def predict_future(model , current_value):
    prediction = [current_value]

    for i in range(5):
        next_value = model.predict(np.array([[prediction[-1]]]))
        prediction.append(next_value[0])
    return prediction[1:]

# Weather Analysis Function
def weather_view():
    city = input('Enter any City Name:')
    current_weather = get_current_weather(city)

    # Loading Historical Data
    historical_data = read_historical_data('') # put the file path of CSV file to get the data

    #prepare and train the Rain prediction model
    x , y , le = prepare_data(historical_data)
    rain_model = train_rain_model(x , y)
    