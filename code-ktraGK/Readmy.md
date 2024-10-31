1. Iris dataset
Bài toán: Phân loại
from sklearn.datasets import load_iris
iris = load_iris()

2.Wine dataset
Bài toán: Phân loại
from sklearn.datasets import load_wine
wine = load_wine()

3. Breast Cancer dataset
Bài toán: Phân loại
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

4. Diabetes dataset
Bài toán: Hồi quy
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()

6. Digits dataset
Bài toán: Phân loại
from sklearn.datasets import load_digits
digits = load_digits()

7. California Housing dataset
Bài toán: Hồi quy
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()

8. Linnerud dataset
Bài toán: Hồi quy đa biến
from sklearn.datasets import load_linnerud
linnerud = load_linnerud()

20 Newsgroups dataset (Phân loại văn bản)
Mô tả: Dữ liệu về các bài viết từ 20 nhóm thảo luận trên Usenet.
from sklearn.datasets import fetch_20newsgroups
newsgroups = fetch_20newsgroups(subset='train')

Olivetti Faces dataset (Phân loại hoặc nhận dạng khuôn mặt)
Mô tả: Hình ảnh khuôn mặt của 40 người khác nhau.
from sklearn.datasets import fetch_olivetti_faces
faces = fetch_olivetti_faces()

LFW People dataset (Nhận dạng khuôn mặt)
Mô tả: Bộ dữ liệu khuôn mặt của những người nổi tiếng.
from sklearn.datasets import fetch_lfw_people
lfw_people = fetch_lfw_people()

LFW Pairs dataset (Phân loại khuôn mặt theo cặp)
Mô tả: Bộ dữ liệu chứa cặp khuôn mặt, mỗi cặp được gán nhãn là cùng người hoặc khác người.
from sklearn.datasets import fetch_lfw_pairs
lfw_pairs = fetch_lfw_pairs()

Covtype dataset (Phân loại)
Mô tả: Phân loại loại thảm thực vật dựa trên thông tin địa lý.
from sklearn.datasets import fetch_covtype
covtype = fetch_covtype()

RCV1 dataset (Phân loại văn bản)
Mô tả: Dữ liệu phân loại văn bản từ Reuters Corpus Volume 1.
from sklearn.datasets import fetch_rcv1
rcv1 = fetch_rcv1()

KDD Cup 99 dataset (Phát hiện xâm nhập mạng)
Mô tả: Bộ dữ liệu về phát hiện xâm nhập mạng từ cuộc thi KDD Cup 1999.
from sklearn.datasets import fetch_kddcup99
kddcup = fetch_kddcup99()

OpenML datasets (Tải các bộ dữ liệu từ OpenML)
Mô tả: Bạn có thể tải hàng ngàn bộ dữ liệu từ OpenML.
from sklearn.datasets import fetch_openml
openml_data = fetch_openml(data_id=your_data_id)

III. Bộ dữ liệu mẫu (Synthetic Datasets)
make_classification (Tạo bộ dữ liệu phân loại tổng hợp)
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=100, n_features=20)

make_regression (Tạo bộ dữ liệu hồi quy tổng hợp)
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=100, n_features=20)

make_blobs (Tạo cụm dữ liệu)
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=100, centers=3)

make_moons và make_circles (Tạo dữ liệu không tuyến tính)
from sklearn.datasets import make_moons, make_circles
X_moons, y_moons = make_moons(n_samples=100)
X_circles, y_circles = make_circles(n_samples=100)

CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT
CRIM: tỷ lệ tội phạm bình quân đầu người theo thị trấn.
ZN: tỷ lệ đất ở được phân vùng cho các lô đất trên 25.000 ft vuông.
INDUS: tỷ lệ mẫu Anh kinh doanh không bán lẻ theo thị trấn.
CHAS: biến giả Sông Charles (= 1 nếu ranh giới khu đất là sông; 0 nếu không).
NOX: nồng độ oxit nitric (phần trên 10 triệu).
RM: số phòng trung bình trên mỗi ngôi nhà.
AGE: tỷ lệ các đơn vị do chủ sở hữu chiếm giữ được xây dựng trước năm 1940.
DIS: khoảng cách có trọng số đến năm trung tâm việc làm của Boston.
RAD: chỉ số khả năng tiếp cận các xa lộ xuyên tâm.
TAX: tỷ lệ thuế tài sản toàn phần trên 10.000 đô la.
PTRATIO: tỷ lệ học sinh-giáo viên theo thị trấn.
B: 1000(Bk – 0,63)^2 trong đó Bk là tỷ lệ người da đen theo thị trấn.
LSTAT: % tình trạng thấp kém của dân số.