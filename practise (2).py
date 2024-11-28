import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import pickle
# load dataset
data = pd.read_csv('co2.csv')
data['time'] = pd.to_datetime(data['time'], yearfirst = True)
data['co2'] = data['co2'].interpolate()


# create a new data has window_size and targer columns

def create_data(data, window_size, target_size):
    i = 1 
    while i < window_size:
        data[f'co2_{i}'] = data['co2'].shift(-i)
        i += 1
    i = 0
    while i < target_size:
        data[f'target_{i}'] = data['co2'].shift(-window_size -i)
        i += 1

    data = data.dropna(axis = 0)
    return data

# choose the window_size and target_size and train_ratio

window_size = 6
target_size = 4
train_ratio = 0.8
targets = [f'target_{i}' for i in range(target_size)]

data = create_data(data, window_size, target_size)
x = data.drop(['time'] + targets, axis = 1)
y = data[targets]


x_train = x[:int(len(data)*train_ratio)]
y_train = y[:int(len(data)*train_ratio)]
x_test = x[int(len(data)*train_ratio):]
y_test = y[int(len(data)*train_ratio):]

models = [LinearRegression() for i in range(target_size)]


r2 = []
MSE = []
for i, model in enumerate(models):
    model.fit(x_train, y_train[f'target_{i}'])
    y_predict = model.predict(x_test)

    r2.append(r2_score(y_test[f'target_{i}'], y_predict))
    MSE.append(mean_squared_error(y_test[f'target_{i}'], y_predict))


    with open(f"model_{i}.pkl", 'wb') as time_series_modles:
        pickle.dump(model, time_series_modles)


print(f'r2: {r2}')
print(f'mean_squaed_error: {MSE}')