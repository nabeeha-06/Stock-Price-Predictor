
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout


# Step 1: Load Dataset

url = "https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv"
data = pd.read_csv(url)

# Select required columns
data = data[['AAPL.Open', 'AAPL.High', 'AAPL.Low', 'AAPL.Close', 'AAPL.Volume']]
data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

print("First 5 rows:")
print(data.head())
print("\nData Shape:", data.shape)


# Step 2: Data Preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

train_size = int(len(scaled_data) * 0.8)

train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Step 3: Create Sequences

def create_dataset(dataset, time_step=60):
    X, Y = [], []

    for i in range(time_step, len(dataset)):
        X.append(dataset[i-time_step:i])
        Y.append(dataset[i, 3])   # Close price

    return np.array(X), np.array(Y)

time_step = 60

X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

print("X_train Shape:", X_train.shape)
print("X_test Shape:", X_test.shape)

# Step 4: Build LSTM Model

model = Sequential()

model.add(
    LSTM(
        units=50,
        return_sequences=True,
        input_shape=(X_train.shape[1], X_train.shape[2])
    )
)
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(25))
model.add(Dense(1))

model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)


# Step 5: Train Model

model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    verbose=1
)

# Step 6: Predictions

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Convert predictions back to original scale
train_full = np.zeros((len(train_predict), 5))
test_full = np.zeros((len(test_predict), 5))

train_full[:, 3] = train_predict[:, 0]
test_full[:, 3] = test_predict[:, 0]

train_predict_actual = scaler.inverse_transform(train_full)[:, 3]
test_predict_actual = scaler.inverse_transform(test_full)[:, 3]

# Step 7: RMSE Calculation

y_test_full = np.zeros((len(y_test), 5))
y_test_full[:, 3] = y_test

y_test_actual = scaler.inverse_transform(y_test_full)[:, 3]

rmse = math.sqrt(
    mean_squared_error(
        y_test_actual,
        test_predict_actual
    )
)

print("\nRMSE:", rmse)

# Step 8: Plot Results

plt.figure(figsize=(12, 6))

plt.plot(
    data['Close'].values,
    label='Actual Price'
)

train_plot = np.empty(len(data))
train_plot[:] = np.nan
train_plot[time_step:len(train_predict_actual)+time_step] = train_predict_actual

test_plot = np.empty(len(data))
test_plot[:] = np.nan

start_index = len(train_predict_actual) + (time_step * 2)
test_plot[start_index:len(data)] = test_predict_actual

plt.plot(
    train_plot,
    label='Train Prediction'
)

plt.plot(
    test_plot,
    label='Test Prediction'
)

plt.title("Apple Stock Price Prediction using LSTM")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.show()

# Step 9: Next Day Prediction

last_60_days = scaled_data[-60:]
last_60_days = last_60_days.reshape(1, 60, 5)

next_price_scaled = model.predict(last_60_days)

next_price_full = np.zeros((1, 5))
next_price_full[:, 3] = next_price_scaled[:, 0]

next_price_actual = scaler.inverse_transform(next_price_full)

print(
    "\nPredicted Next Day Closing Price:",
    round(next_price_actual[0, 3], 2)
)
