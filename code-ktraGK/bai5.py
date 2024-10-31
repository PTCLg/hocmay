# Bài tập 5: Dự đoán lượng mưa

import pandas as pd
from sklearn.model_selection import train_test_split

# Tải dữ liệu thời tiết
data = pd.read_csv('dataset/weather.csv')

# Chia dữ liệu thành tập đặc trưng (X) và nhãn (y - lượng mưa)
X = data.drop(columns=['RISK_MM'])  # Loại bỏ cột 'RISK_MM' để lấy các yếu tố khí hậu
y = data['RISK_MM']

# Chia dữ liệu thành tập huấn luyện (80%) và kiểm tra (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Xử lý giá trị thiếu (thay thế các giá trị thiếu bằng trung bình)
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from huanluyen import RandomForestRegressor

# Áp dụng Random Forest Regression
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred_rf = rf.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Đánh giá mô hình Random Forest
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest - MAE: {mae_rf:.2f}, MSE: {mse_rf:.2f}, R²: {r2_rf:.2f}")

from huanluyen import LinearRegression

# Áp dụng Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred_lr = lr.predict(X_test)

# Đánh giá mô hình Linear Regression
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f"Linear Regression - MAE: {mae_lr:.2f}, MSE: {mse_lr:.2f}, R²: {r2_lr:.2f}")
