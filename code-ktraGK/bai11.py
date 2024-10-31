# Bài tập 11: Dự đoán bệnh tim

import pandas as pd
from sklearn.model_selection import train_test_split

# Tải dữ liệu bệnh tim
data = pd.read_csv('dataset/heart_disease_data.csv')

# Chia dữ liệu thành đặc trưng (X) và nhãn (y)
X = data.drop('target', axis=1)  # Giả sử cột 'target' là nhãn
y = data['target']

# Chia dữ liệu thành tập huấn luyện (80%) và kiểm tra (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Xử lý giá trị thiếu
imputer = SimpleImputer(strategy='mean')  # Hoặc strategy='median', 'most_frequent', v.v.
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

from huanluyen import LogisticRegression

# Khởi tạo mô hình Logistic Regression
logistic_model = LogisticRegression()

# Huấn luyện mô hình 
logistic_model.fit(X_train_scaled, y_train)

# Dự đoán trên tập kiểm tra
y_pred_logistic = logistic_model.predict(X_test_scaled)

from sklearn.metrics import accuracy_score, f1_score

# Tính độ chính xác và F1-score
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
f1_logistic = f1_score(y_test, y_pred_logistic)

print(f"Logistic Regression - Accuracy: {accuracy_logistic:.2f}, F1-score: {f1_logistic:.2f}")

from huanluyen import RandomForestClassifier

# Khởi tạo mô hình Random Forest
rf_model = RandomForestClassifier()

# Huấn luyện mô hình
rf_model.fit(X_train_scaled, y_train)

# Dự đoán trên tập kiểm tra
y_pred_rf = rf_model.predict(X_test_scaled)

# Tính độ chính xác và F1-score cho Random Forest
accuracy_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

print(f"Random Forest - Accuracy: {accuracy_rf:.2f}, F1-score: {f1_rf:.2f}")

