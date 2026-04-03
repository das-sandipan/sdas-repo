import requests # Lib to fetch data from API
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # to split data intontraining and test data set
from sklearn.preprocessing import LabelEncoder # to convert categorical value to numerical value
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor # for clasiification and regression
from sklearn.metrics import mean_squared_error # to measure accuracy of prediction model
from datetime import datetime, timedelta # for using date and time
import pytz 

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
        'wind_gust_dir': data['wind']['deg'],   # API specific code
        'pressure': data['main']['pressure'],   # API specific code
        'Wind_Gust_Speed': data['wind']['speed'], # API specific code
    }
# Read Historical Data 
def get_historical_data(filename):
    df = pd.read_csv(filename) #load CSV file into dataframe
    df = df.dropna() # to remove "NA" value 
    df = df.drop_duplicates() # to remove duplicates
    return df

# Prepare Data for Training
def prepare_data (data):
    le = LabelEncoder() # create label encoder instance
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
    city = input('Enter any City Name: ')
    current_weather = get_current_weather(city)

    # Loading Historical Data
    historical_data = read_historical_data('') # put the file path of CSV file to get the data

    #prepare and train the Rain prediction model
    x , y , le = prepare_data(historical_data)
    rain_model = train_rain_model(x , y)

    # Map Wind Direction to Compass Points
    wind_deg = current_weather['wind_gust_dir'] % 360
    compass_points = [
        ("N", 0, 11.25), ("NNE", 11.25, 33.75), ("NE", 33.75, 56.25),
        ("ENE", 56.25, 78.75), ("E", 78.75, 101.25), ("ESE", 101.25, 123.75),
        ("SE", 123.75, 146.25), ("SSE", 146.25, 168.75), ("S", 168.75, 191.25),
        ("SSW", 191.25, 213.75), ("SW", 213.75, 236.25), ("WSW", 236.25, 258.75),
        ("W", 258.75, 281.25), ("WNW", 281.25, 303.75), ("NW", 303.75, 326.25),
        ("NNW", 326.25, 348.75)
    ]

    compass_direction = next(point for point, start , end in compass_points if start <= wind_deg < end)

    compass_direction_encoded = le.transform([compass_direction])[0] if compass_direction in le.classes_ else -1

    current_data = {
        'MinTemp': current_weather('temp_min'),
        'MaxTemp': current_weather('temp_max'),
        'WindGustDir': compass_direction_encoded,
        'WindGustSpeed': current_weather('Wind_Gust_Speed'),
        'Humidity': current_weather('humidity'),
        'Pressure': current_weather('pressure'),
        'Temp': current_weather('current_temp'),
    }

    current_df = pd.DataFrame([current_data])

    # Rain Prediction
    rain_prediction = rain_model.predict(current_df)[0]

    # Prep Regression model for Temp & Humidity
    x_temp, y_temp = prepare_regression_data(historical_data, 'Temp')
    x_hum, y_hum = prepare_regression_data(historical_data, 'Humidity')
    temp_model = train_regression_model(x_temp, y_temp)
    hum_model = train_regression_model(x_hum, y_hum)

    # Predict Future Temp & Humidity
    future_temp = predict_future(temp_model, current_weather['temp_min'])
    future_humidity = predict_future(hum_model, current_weather['humidity'])

    # Prep Time for future prediction
    timezone = pytz.timezone('Asia/India')
    now = datetime.now(timezone)
    next_hour = now + timedelta(hours=1)
    next_hour = next_hour.replace(minute=0, second=0, microsecond=0)

    future_time = [(next_hour + timedelta(hours=1)).strftime("%H:00") for i in range(5)]

    # Display Result
    print(f"City: {city}, {current_weather['country']}")
    print(f"Current Temperature: {current_weather['current_temp']}")
    print(f"Feels Like: {current_weather['feels_like']}")
    print(f"Minimum Temperature: {current_weather['temp_min']}°C")
    print(f"Maximum Temperature: {current_weather['temp_max']}°C")
    print(f"Humidity: {current_weather['humidity']}%")
    print(f"Weather Prediction: {current_weather['description']}")
    print(f"Rain Prediction: {'Yes' if rain_prediction else 'No'}")

    print("\nFuture Temperature Prediction:")
    for time, temp in zip(future_time, future_temp):
        print(f"{time}: {round(temp, 1)}°C")
    
    print("\nFuture Humidity Prediction")
    for time, humidity in zip(future_time, future_humidity):
        print(f"{time}: {round(humidity, 1)}%")
        
weather_view()





    