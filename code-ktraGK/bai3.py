# Bài tập 3: Phân loại bệnh tiểu đường

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from huanluyen import SVC, LogisticRegression
# from sklearn.svm import SVC

# Tải dữ liệu tiểu đường
diabetes = load_diabetes(as_frame=True)
X = diabetes.data
# Chuyển đổi thành bài toán phân loại (giá trị > 140 coi là mắc bệnh)
y = (diabetes.target > 140).astype(int)

# Chia dữ liệu thành tập huấn luyện (80%) và tập kiểm tra (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Áp dụng mô hình Logistic Regression
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred_log = log_reg.predict(X_test)

# Tính các chỉ số đánh giá
accuracy_log = accuracy_score(y_test, y_pred_log)
precision_log = precision_score(y_test, y_pred_log)
recall_log = recall_score(y_test, y_pred_log)
f1_log = f1_score(y_test, y_pred_log)

print(f"Logistic Regression - Độ chính xác: {accuracy_log:.2f}, Độ chính xác (Precision): {precision_log:.2f}, "
      f"Độ nhạy (Recall): {recall_log:.2f}, F1-score: {f1_log:.2f}")

# Áp dụng mô hình SVM
svm = SVC(random_state=42)
svm.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred_svm = svm.predict(X_test)

# Tính các chỉ số đánh giá cho SVM
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)

print(f"SVM - Độ chính xác: {accuracy_svm:.2f}, Độ chính xác (Precision): {precision_svm:.2f}, "
      f"Độ nhạy (Recall): {recall_svm:.2f}, F1-score: {f1_svm:.2f}")
