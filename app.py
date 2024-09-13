# app.py

import os
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
import numpy as np
import joblib
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'  # Thay đổi đường dẫn

# Tạo thư mục nếu chưa có
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Tải mô hình
svm_model = joblib.load('models/svm_shoe_classifier_model.pkl')
mlp_model = joblib.load('models/mlp_shoe_classifier_model.pkl')

# Tải scaler
scaler = joblib.load('models/scaler.pkl')

# Tạo mô hình VGG16 cho việc trích xuất đặc trưng
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
img_size = (224, 224)

# Các loại giày dép
categories = ['boots', 'dressShoes', 'sandals', 'sneakers']

# Hàm xử lý và phân loại ảnh
def extract_features(img_path):
    img = image.load_img(img_path, target_size=img_size)
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = model.predict(img_data)
    return features.flatten()

def classify_image(img_path):
    features = extract_features(img_path)
    
    # Tiền xử lý dữ liệu
    features = features.reshape(1, -1)  # Đảm bảo dữ liệu có dạng 2D
    features = scaler.transform(features)
    
    # Phân loại bằng mô hình SVM
    svm_prediction = svm_model.predict(features)[0]
    svm_label = categories[svm_prediction]
    
    # Phân loại bằng mô hình MLP
    mlp_prediction = mlp_model.predict(features)[0]
    mlp_label = categories[np.argmax(mlp_prediction)]
    
    return svm_label, mlp_label

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            svm_label, mlp_label = classify_image(filepath)
            
            return render_template('index.html', filename=filename, svm_label=svm_label, mlp_label=mlp_label)
    
    return render_template('index.html', filename=None, svm_label=None, mlp_label=None)

if __name__ == '__main__':
    app.run(debug=True)
