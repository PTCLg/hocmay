# Bài tập 10: Phân tích cảm xúc trên đánh giá sản phẩm

# import random
# import pandas as pd

# # Danh sách các từ khóa tích cực và tiêu cực
# positive_keywords = [
#     "tuyệt vời", "yêu thích", "tuyệt hảo", "hoàn hảo", "sang trọng",
#     "thú vị", "xuất sắc", "hài lòng", "dễ sử dụng", "đáng giá", "rất hài lòng", "khuyên bạn nên mua", "sẽ mua lại"
# ]

# negative_keywords = [
#     "kém", "thất vọng", "không đáng", "tệ", "xấu",
#     "khó sử dụng", "kém chất lượng", "mất tiền", "không hài lòng", "không tốt"
# ]

# # Hàm tạo câu đánh giá từ từ khóa
# def generate_review(keywords, sentiment):
#     if sentiment == "positive":
#         return f"{random.choice(keywords)}"
#     else:
#         return f"{random.choice(keywords)}"

# # Tạo tập dữ liệu với số lượng lớn
# def create_reviews_dataset(num_reviews):
#     reviews = []
#     sentiments = []
    
#     for _ in range(num_reviews // 2):  # Tạo nửa số đánh giá tích cực
#         review = generate_review(positive_keywords, "positive")
#         reviews.append(review)
#         sentiments.append(1)  # Gán nhãn 1 cho cảm xúc tích cực
    
#     for _ in range(num_reviews // 2):  # Tạo nửa số đánh giá tiêu cực
#         review = generate_review(negative_keywords, "negative")
#         reviews.append(review)
#         sentiments.append(0)  # Gán nhãn 0 cho cảm xúc tiêu cực
    
#     # Tạo DataFrame
#     df = pd.DataFrame({
#         'Review': reviews,
#         'Sentiment': sentiments
#     })
    
#     # Shuffle để trộn đánh giá tích cực và tiêu cực
#     df = df.sample(frac=1).reset_index(drop=True)
    
#     return df

# # Tạo tập dữ liệu với 1000 đánh giá
# num_reviews = 500
# df = create_reviews_dataset(num_reviews)

# # Lưu ra file CSV
# df.to_csv('large_product_reviews_with_keywords.csv', index=False)

# print(f"Tập dữ liệu đã tạo với {num_reviews} đánh giá.")
# print(df.head())

import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from huanluyen import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score

# 1. Tải dữ liệu từ file CSV
df = pd.read_csv('dataset/product_reviews.csv')

# 2. Chia dữ liệu thành tập huấn luyện (80%) và tập kiểm tra (20%)
X_train, X_test, y_train, y_test = train_test_split(df['Review'], df['Sentiment'], test_size=0.2, random_state=42)

print("Tập huấn luyện:", len(X_train), "đánh giá")
print("Tập kiểm tra:", len(X_test), "đánh giá")

# 3. Tiền xử lý văn bản: xóa dấu câu và chuyển đổi về chữ thường
def preprocess_text(text):
    text = text.lower()  # Chuyển về chữ thường
    text = text.translate(str.maketrans('', '', string.punctuation))  # Xóa dấu câu
    return text

# Áp dụng tiền xử lý
X_train_processed = X_train.apply(preprocess_text)
X_test_processed = X_test.apply(preprocess_text)

# 4. Sử dụng TF-IDF để chuyển đổi văn bản thành dạng số
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train_processed)
X_test_tfidf = tfidf.transform(X_test_processed)

print("Kích thước tập huấn luyện sau TF-IDF:", X_train_tfidf.shape)
print("Kích thước tập kiểm tra sau TF-IDF:", X_test_tfidf.shape)

# 5. Áp dụng thuật toán Naive Bayes để xây dựng mô hình phân loại
model = MultinomialNB(alpha=0.001)  # Khởi tạo mô hình Naive Bayes
model.fit(X_train_tfidf, y_train)  # Huấn luyện mô hình

# 6. Đánh giá mô hình bằng độ chính xác và F1-score
y_pred = model.predict(X_test_tfidf)  # Dự đoán trên tập kiểm tra

# Tính toán độ chính xác và F1-score
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Độ chính xác: {accuracy:.2f}")
print(f"F1-score: {f1:.2f}")

# 7. Đề xuất phương pháp cải thiện dự đoán
print("\nĐề xuất cải thiện dự đoán:")
print("- Tăng cường dữ liệu: Tạo thêm dữ liệu đánh giá.")
print("- Sử dụng mô hình phức tạp hơn như Logistic Regression hoặc mạng nơ-ron.")
print("- Tuning Hyperparameters: Tối ưu hóa tham số của mô hình.")
print("- Sử dụng kỹ thuật chọn đặc trưng: Chọn những từ quan trọng nhất.")
print("- Thực hiện xử lý ngôn ngữ tự nhiên nâng cao: Sử dụng stemming hoặc lemmatization.")
