# Bài tập 9: Dự đoán sự hài lòng của khách hàng

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
# from sklearn.ensemble import RandomForestClassifier
from huanluyen import RandomForestClassifier
# from sklearn.svm import SVC
from huanluyen import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer

# Bước 1: Tải dữ liệu và chia tập huấn luyện (80%) và tập kiểm tra (20%)
df = pd.read_csv('dataset/customer_satisfaction.csv')

# Chia dữ liệu thành đặc trưng (X) và mục tiêu (y)
X = df.drop('satisfaction', axis=1)
y = df['satisfaction']

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Bước 2: Tiền xử lý dữ liệu
# Các cột phân loại và số
categorical_cols = ['Customer Type', 'Type of Travel', 'Class']
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Bộ tiền xử lý cho dữ liệu
preprocessor = ColumnTransformer(transformers=[
    ('num', SimpleImputer(strategy='mean'), numeric_cols),  # Xử lý NaN cho cột số bằng giá trị trung bình
    ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),  # Xử lý NaN cho cột phân loại bằng giá trị phổ biến nhất
                           ('onehot', OneHotEncoder())]), categorical_cols)], remainder='passthrough')

# Áp dụng bộ tiền xử lý lên dữ liệu huấn luyện và kiểm tra
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Chuẩn hóa các cột số nếu cần
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Bước 3: Xây dựng mô hình với thuật toán Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)  # Huấn luyện mô hình
y_pred_rf = rf_model.predict(X_test)  # Dự đoán trên tập kiểm tra

# Bước 4: Đánh giá hiệu quả của mô hình Random Forest
accuracy_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')

print(f"Random Forest Accuracy: {accuracy_rf * 100:.2f}%")
print(f"Random Forest F1-score: {f1_rf:.2f}")

# Bước 5: So sánh với thuật toán SVM
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)  # Huấn luyện mô hình SVM
y_pred_svm = svm_model.predict(X_test)  # Dự đoán trên tập kiểm tra

# Đánh giá hiệu quả của mô hình SVM
accuracy_svm = accuracy_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm, average='weighted')

print(f"SVM Accuracy: {accuracy_svm * 100:.2f}%")
print(f"SVM F1-score: {f1_svm:.2f}")

# Bước 6: Giải thích sự khác biệt và lựa chọn mô hình tối ưu
if accuracy_rf > accuracy_svm:
    print("Random Forest is the better model based on accuracy.")
else:
    print("SVM is the better model based on accuracy.")
