# create_dataset.py

import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# Khởi tạo mô hình VGG16 với trọng số ImageNet
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

# Đường dẫn đến thư mục dữ liệu
data_dir = 'dataset/shoes/'
non_shoes_dir = 'dataset/non_shoes/'  # Thêm thư mục chứa ảnh không phải giày

# Các loại giày dép và nhãn không phải giày dép
categories = ['boots', 'dressShoes', 'sandals', 'sneakers']
binary_categories = ['shoes', 'non_shoes']

# Kích thước hình ảnh đầu vào cho mô hình VGG16
img_size = (224, 224)

# Danh sách lưu đặc trưng và nhãn
features_list = []
labels_list = []

# Danh sách lưu đặc trưng nhị phân (giày vs không phải giày)
binary_features_list = []
binary_labels_list = []

# Hàm xử lý từng hình ảnh
def extract_features(img_path):
    print(f"Processing file: {img_path}")  # In ra đường dẫn tệp để kiểm tra
    img = image.load_img(img_path, target_size=img_size)
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = model.predict(img_data)
    return features.flatten()

# Duyệt qua từng loại giày và hình ảnh trong thư mục
for category in categories:
    folder_path = os.path.join(data_dir, category)
    label = categories.index(category)
    
    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        
        # Trích xuất đặc trưng
        features = extract_features(img_path)
        
        # Lưu đặc trưng và nhãn vào danh sách
        features_list.append(features)
        labels_list.append(label)

        # Lưu đặc trưng cho mô hình nhị phân (là giày)
        binary_features_list.append(features)
        binary_labels_list.append(1)

# Thêm dữ liệu không phải giày dép
for img_file in os.listdir(non_shoes_dir):
    img_path = os.path.join(non_shoes_dir, img_file)
    
    # Trích xuất đặc trưng
    features = extract_features(img_path)
    
    # Lưu đặc trưng cho mô hình nhị phân (không phải giày)
    binary_features_list.append(features)
    binary_labels_list.append(0)

# Chuyển đổi danh sách thành DataFrame
df = pd.DataFrame(features_list)
df['label'] = labels_list

binary_df = pd.DataFrame(binary_features_list)
binary_df['label'] = binary_labels_list

# Lưu tập dữ liệu ra file CSV
df.to_csv('shoe_features_dataset.csv', index=False)
binary_df.to_csv('shoe_vs_non_shoe_dataset.csv', index=False)

print("Hoàn thành trích xuất đặc trưng và lưu")
