import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression


# Load dataset
data = pd.read_csv('co2.csv')
data['time'] = pd.to_datetime(data['time'], yearfirst = True)
data['co2'] = data['co2'].interpolate()


# create function for window and target

def create_data(data, window_size, target_size):
    i = 1
    while i < window_size:
        data[f'co2_{i}'] = data['co2'].shift(-i)
        i += 1
    i = 0
    while i < target_size:
        data[f'target_{i}'] = data['co2'].shift(-i - window_size)
        i += 1

    data = data.dropna(axis = 0)
    return data

window_size = 5
target_size = 3
train_ratio = 0.8
data = create_data(data, window_size, target_size)

targets = [f'target_{i}' for i in range(target_size)]
x = data.drop(['time'] + targets, axis = 1)
y = data[targets]
num_samples = len(y)

x_train = x[:int(num_samples * train_ratio)]
y_train = y[:int(len(data)* train_ratio)]
x_test = x[int(len(data)* train_ratio):] 
y_test = y[int(len(data)* train_ratio):] 


models = [LinearRegression() for i in range(target_size)]
r2 =[]
for i, model in enumerate(models):
    model.fit(x_train, y_train[f'target_{i}'])
    y_predict = model.predict(x_test)
    r2.append(r2_score(y_test[f'target_{i}'], y_predict))

print(r2)


y_predict = models[0].predict(x_test)
for i in range(1, 10):
    print(f'True value {y_test[f"target_0"].iloc[-i]}. Predict_value{y_predict[-i]}')