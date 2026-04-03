import requests # Lib to fetch data from API
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # to split data intontraining and test data set
from sklearn.preprocessing import LabelEncoder # to convert categorical value to numerical value
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor # for clasiification and regression
from sklearn.metrics import mean_squared_error # to measure accuracy of prediction model
from datetime import datetime, timedelta # for using date and time
