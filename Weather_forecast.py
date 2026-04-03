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
