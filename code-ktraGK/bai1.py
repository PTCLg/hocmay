# Bài tập 1: Phân loại hoa Iris

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from huanluyen import KNeighborsClassifier

# Tải dữ liệu Iris
iris = load_iris()
X = iris.data
y = iris.target

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

def knn_model(k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Kiểm tra với các giá trị k khác nhau
k_val = range(1, 101)
accs = [knn_model(k) for k in k_val]

# In độ chính xác cho từng giá trị k
for k, acc in zip(k_val, accs):
    print(f'k = {k}: Độ chính xác = {acc:.3f}')

# Vẽ biểu đồ độ chính xác so với k
plt.plot(k_val, accs, marker='o')
plt.xlabel('k')
plt.ylabel('Độ chính xác')
plt.title('Độ chính xác của KNN theo k')
plt.show()
