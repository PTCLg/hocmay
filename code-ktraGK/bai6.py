# Bài tập 6: Phân loại chữ số viết tay (MNIST)

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Tải dữ liệu MNIST
mnist = fetch_openml('mnist_784', version=1)

# Chia dữ liệu thành tập huấn luyện (80%) và kiểm tra (20%)
X = mnist.data
y = mnist.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu (giá trị pixel từ 0-255 thành 0-1)
X_train = X_train / 255.0
X_test = X_test / 255.0

from sklearn.svm import SVC

# Xây dựng mô hình SVM
svm_model = SVC(C=10, gamma='scale', kernel='linear')

# Huấn luyện mô hình
svm_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred_svm = svm_model.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Đánh giá mô hình SVM
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"Độ chính xác của mô hình SVM: {accuracy_svm:.2f}")

# Vẽ ma trận nhầm lẫn cho SVM
cm_svm = confusion_matrix(y_test, y_pred_svm)
disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm)
disp_svm.plot(cmap=plt.cm.Blues)
plt.show()

from sklearn.neighbors import KNeighborsClassifier

# Áp dụng mô hình KNN
knn_model = KNeighborsClassifier(n_neighbors=3)

# Huấn luyện mô hình
knn_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred_knn = knn_model.predict(X_test)

# Đánh giá mô hình KNN
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"Độ chính xác của mô hình KNN: {accuracy_knn:.2f}")

# Vẽ ma trận nhầm lẫn cho KNN
cm_knn = confusion_matrix(y_test, y_pred_knn)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn)
disp_knn.plot(cmap=plt.cm.Blues)
plt.show()