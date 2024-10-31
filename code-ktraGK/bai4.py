# Bài tập 4: Phân loại thư rác

import numpy as np
import pandas as pd
import nltk
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from huanluyen import MultinomialNB

# Bước 1: Tải tập dữ liệu
data = pd.read_csv('dataset/spam_email_dataset.csv')

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(
    data['message'],
    data['label'],
    test_size=0.2,
    random_state=42
)

# Bước 2: Tiền xử lý văn bản
stop_words = set(nltk.corpus.stopwords.words('english'))


def preprocess_text(text):
    # Chuyển sang chữ thường
    text = text.lower()
    # Tokenization
    tokens = nltk.word_tokenize(text)
    # Loại bỏ dấu câu và stop words
    tokens = [
        word for word in tokens if word not in stop_words and word not in string.punctuation]
    return ' '.join(tokens)


# Áp dụng tiền xử lý
X_train = X_train.apply(preprocess_text)
X_test = X_test.apply(preprocess_text)

# Bước 3: Biến đổi văn bản thành dạng số bằng phương pháp TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Bước 4: Áp dụng thuật toán Naive Bayes để phân loại
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Bước 5: Đánh giá độ chính xác và vẽ biểu đồ ma trận nhầm lẫn
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Vẽ ma trận nhầm lẫn
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
