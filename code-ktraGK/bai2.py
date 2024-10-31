# Bài tập 2: Dự đoán giá nhà ở Boston

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.tree import DecisionTreeRegressor
import pandas as pd

from huanluyen import DecisionTreeRegressor, LinearRegression

# Tải dữ liệu Boston
boston = pd.read_csv("dataset/boston_housing.csv")

X = boston.drop(columns=['target'])  # Tất cả các cột trừ 'target'
y = boston['target']  # Cột 'target'

# Chia dữ liệu thành 80% tập huấn luyện và 20% tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Áp dụng mô hình hồi quy tuyến tính
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred_lin = lin_reg.predict(X_test)

# Tính toán các chỉ số đánh giá
mae_lin = mean_absolute_error(y_test, y_pred_lin)
mse_lin = mean_squared_error(y_test, y_pred_lin)
r2_lin = r2_score(y_test, y_pred_lin)

print(
    f"Linear Regression - MAE: {mae_lin:.2f}, MSE: {mse_lin:.2f}, R²: {r2_lin:.2f}")

# Áp dụng mô hình Decision Tree Regression
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred_tree = tree_reg.predict(X_test)

# Đánh giá mô hình Decision Tree Regression
mae_tree = mean_absolute_error(y_test, y_pred_tree)
mse_tree = mean_squared_error(y_test, y_pred_tree)
r2_tree = r2_score(y_test, y_pred_tree)

print(
    f"Decision Tree Regression - MAE: {mae_tree:.2f}, MSE: {mse_tree:.2f}, R²: {r2_tree:.2f}")
