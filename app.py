# app.py

from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import numpy as np
import joblib
from PIL import Image
import os
import io
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Đảm bảo thư mục uploads tồn tại
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Tải mô hình
svm_model = joblib.load('models/svm_shoe_classifier_model.pkl')
mlp_model = joblib.load('models/mlp_shoe_classifier_model.pkl')

# Tải mô hình nhị phân
binary_model = joblib.load('models/binary_shoe_classifier_model.pkl')

# Tải scaler
scaler = joblib.load('models/scaler.pkl')
binary_scaler = joblib.load('models/binary_scaler.pkl')

# Tạo mô hình VGG16 cho việc trích xuất đặc trưng
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
img_size = (224, 224)

# Các loại giày dép
categories = ['boots', 'dressShoes', 'sandals', 'sneakers']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_features(img_path):
    try:
        img = Image.open(img_path)
        img = img.convert('RGB')  # Đảm bảo ảnh ở định dạng RGB
        img = img.resize(img_size)
        img_data = np.array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        features = model.predict(img_data)
        return features.flatten()
    except Exception as e:
        print(f"Error processing file {img_path}: {e}")
        return None

def classify_image(img_path):
    features = extract_features(img_path)
    
    if features is None:
        return None, None, None
    
    # Tiền xử lý dữ liệu cho phân loại nhị phân
    features_bin = features.reshape(1, -1)
    features_bin = binary_scaler.transform(features_bin)
    
    # Phân loại bằng mô hình nhị phân
    binary_prediction = binary_model.predict(features_bin)[0]
    
    if binary_prediction == 0:  # Không phải giày
        return None, None, "This image is not classified as a shoe."
    
    # Phân loại bằng mô hình SVM
    features = features.reshape(1, -1)  # Đảm bảo dữ liệu có dạng 2D
    features = scaler.transform(features)
    
    svm_prediction = svm_model.predict(features)[0]
    svm_label = categories[svm_prediction]
    
    # Phân loại bằng mô hình MLP
    mlp_prediction = mlp_model.predict(features)[0]
    mlp_label = categories[np.argmax(mlp_prediction)]
    
    return svm_label, mlp_label, None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error_message='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error_message='No selected file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            svm_label, mlp_label, error_message = classify_image(filepath)
            
            if error_message:
                return render_template('index.html', filename=filename, error_message=error_message)
            
            return render_template('index.html', filename=filename, svm_label=svm_label, mlp_label=mlp_label)
        else:
            return render_template('index.html', error_message='Invalid file format. Please upload a PNG, JPG, JPEG, or GIF image.')
    
    return render_template('index.html', filename=None, svm_label=None, mlp_label=None, error_message=None)

if __name__ == '__main__':
    app.run(debug=True)
