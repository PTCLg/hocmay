# train_model.py

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle

# Đọc tập dữ liệu đặc trưng từ file CSV
df = pd.read_csv('shoe_features_dataset.csv')

# Tách dữ liệu thành đặc trưng (X) và nhãn (y)
X = df.drop('label', axis=1)
y = df['label']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra (80% huấn luyện, 20% kiểm tra)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tiền xử lý dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Lưu scaler
with open('models/scaler.pkl', 'wb') as f:
    joblib.dump(scaler, f)

# Sử dụng SVM (Support Vector Machine)
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Dự đoán trên tập kiểm tra
y_pred_svm = svm_model.predict(X_test_scaled)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"Độ chính xác của mô hình SVM: {accuracy_svm * 100:.2f}%")

# Sử dụng MLPClassifier (Multi-Layer Perceptron)
mlp_model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, activation='relu', solver='adam', random_state=42)
mlp_model.fit(X_train_scaled, y_train)

# Dự đoán trên tập kiểm tra
y_pred_mlp = mlp_model.predict(X_test_scaled)
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
print(f"Độ chính xác của mô hình MLP: {accuracy_mlp * 100:.2f}%")

# Lưu mô hình SVM
with open('models/svm_shoe_classifier_model.pkl', 'wb') as f:
    pickle.dump(svm_model, f)

# Lưu mô hình MLP
with open('models/mlp_shoe_classifier_model.pkl', 'wb') as f:
    pickle.dump(mlp_model, f)

print("Mô hình SVM và MLP đã được lưu thành công bằng pickle.")

# Huấn luyện mô hình nhị phân
df_binary = pd.read_csv('shoe_vs_non_shoe_dataset.csv')
X_binary = df_binary.drop('label', axis=1)
y_binary = df_binary['label']

X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(X_binary, y_binary, test_size=0.2, random_state=42)

scaler_bin = StandardScaler()
X_train_bin_scaled = scaler_bin.fit_transform(X_train_bin)
X_test_bin_scaled = scaler_bin.transform(X_test_bin)

binary_model = LogisticRegression(random_state=42)
binary_model.fit(X_train_bin_scaled, y_train_bin)

with open('models/binary_shoe_classifier_model.pkl', 'wb') as f:
    pickle.dump(binary_model, f)

with open('models/binary_scaler.pkl', 'wb') as f:
    joblib.dump(scaler_bin, f)

print("Mô hình nhị phân (giày dép vs không phải giày dép) đã được lưu.")