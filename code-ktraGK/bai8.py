# Bài tập 8: Dự đoán điểm số sinh viên

# # Tạo lại dataset với công thức tính điểm đã điều chỉnh

# # Thiết lập số lượng sinh viên
from huanluyen import DecisionTreeRegressor, LassoRegression, RidgeRegression, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# data = {
#     'hours': [2.5, 5.1, 3.2, 8.5, 3.5, 1.5, 9.2, 5.5, 8.3, 2.7,
#                       7.7, 5.9, 4.5, 3.3, 1.1, 8.9, 2.5, 1.9, 6.1, 7.4,
#                       2.7, 4.8, 3.8, 6.9, 7.8],
#     'activity_level': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
#                        20, 30, 40, 50, 60, 70, 10, 20, 30, 40,
#                        50, 60, 70, 80, 90]  # Đảm bảo rằng số lượng giá trị khớp với hours_studied
# }

# # Tạo DataFrame
# df = pd.DataFrame(data)

# # Tính toán điểm số cuối kỳ dựa trên các đặc trưng
# max_study_hours = 10  # Giả sử số giờ học tối đa là 10
# max_activity_participation = 100  # Giả sử mức độ tham gia tối đa là 100

# # Sử dụng một công thức để tính điểm số cuối kỳ
# df['score'] = (df['hours'] / max_study_hours * 40) + \
#               (df['activity_level'] / max_activity_participation * 60)

# # Đảm bảo rằng score không vượt quá 100
# df['score'] = df['score'].clip(0, 100)

# # Lưu dataset vào file CSV
# csv_file_path = 'student_scores.csv'
# df.to_csv(csv_file_path, index=False)

# print("Dataset đã được lưu thành công vào file:", csv_file_path)


from sklearn.model_selection import train_test_split

# Tải dữ liệu điểm số sinh viên
data = pd.read_csv('dataset/student_scores.csv')

# Chia tập dữ liệu thành đặc trưng (X) và nhãn (y)
X = data[['hours']]
y = data['score']

# Chia thành tập huấn luyện (80%) và kiểm tra (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Xây dựng mô hình hồi quy tuyến tính với các tham số điều chỉnh
lr_model = LinearRegression()

# Huấn luyện mô hình
lr_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred_lr = lr_model.predict(X_test)

# Tính các chỉ số đánh giá
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(
    f"MAE: {mae_lr:.2f}, MSE: {mse_lr:.2f}, Rs: {r2_lr:.2f}")

# Sử dụng LASSO regularization
lasso_model = LassoRegression(alpha=0.1)  # Alpha có thể điều chỉnh
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)

# Tính các chỉ số đánh giá cho LASSO
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)
print(f"LASSO Regression - MAE: {mae_lasso:.2f}, MSE: {
      mse_lasso:.2f}, R-squared: {r2_lasso:.2f}")

# Sử dụng Ridge regularization
ridge_model = RidgeRegression(alpha=0.1)  # Alpha có thể điều chỉnh
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)

# Tính các chỉ số đánh giá cho Ridge
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)
print(f"Ridge Regression - MAE: {mae_ridge:.2f}, MSE: {
      mse_ridge:.2f}, R-squared: {r2_ridge:.2f}")


# Xây dựng mô hình Decision Tree Regression
dt_model = DecisionTreeRegressor(random_state=42)

# Huấn luyện mô hình
dt_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred_dt = dt_model.predict(X_test)

# Đánh giá mô hình Decision Tree
mae_dt = mean_absolute_error(y_test, y_pred_dt)
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

print(
    f"MAE: {mae_dt:.2f}, MSE: {mse_dt:.2f}, Rs: {r2_dt:.2f}")
