# Bài tập 7: Dự đoán giá cổ phiếu

import pandas as pd
from sklearn.model_selection import train_test_split

# Tải dữ liệu giá cổ phiếu
data = pd.read_csv('dataset/stock_data.csv')

# Chọn các cột cần thiết để dự đoán, bỏ cột 'Date'
X = data.drop(columns=['Date', 'Name', 'Close'])
y = data['Close']

# Chia dữ liệu thành tập huấn luyện (80%) và kiểm tra (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler

# Xử lý giá trị thiếu
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from huanluyen import LinearRegression

# Xây dựng mô hình hồi quy tuyến tính
lr_model = LinearRegression()

# Huấn luyện mô hình
lr_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred_lr = lr_model.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Tính các chỉ số đánh giá
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f"Linear Regression - MAE: {mae_lr:.2f}, MSE: {mse_lr:.2f}, R-squared: {r2_lr:.2f}")

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Chuyển đổi dữ liệu cho mô hình LSTM
def create_sequences(X, y, time_steps=60):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i+time_steps])
        ys.append(y[i+time_steps])
    return np.array(Xs), np.array(ys)

# Chuẩn bị dữ liệu cho LSTM
time_steps = 60
X_train_lstm, y_train_lstm = create_sequences(X_train, y_train.values, time_steps)
X_test_lstm, y_test_lstm = create_sequences(X_test, y_test.values, time_steps)

# Xây dựng mô hình LSTM
lstm_model = Sequential()
lstm_model.add(LSTM(50, return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
lstm_model.add(LSTM(50))
lstm_model.add(Dense(1))

# Compile mô hình
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Huấn luyện mô hình
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=32)

# Dự đoán trên tập kiểm tra
y_pred_lstm = lstm_model.predict(X_test_lstm)

# Tính toán các chỉ số cho LSTM
mae_lstm = mean_absolute_error(y_test_lstm, y_pred_lstm)
mse_lstm = mean_squared_error(y_test_lstm, y_pred_lstm)
r2_lstm = r2_score(y_test_lstm, y_pred_lstm)

print(f"LSTM - MAE: {mae_lstm:.2f}, MSE: {mse_lstm:.2f}, R-squared: {r2_lstm:.2f}")
