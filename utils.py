# utils.py

import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import joblib
import os

# Load pre-trained VGG16 model for feature extraction
vgg_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
img_size = (224, 224)

def extract_features(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=img_size)
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    
    # Extract features using VGG16
    features = vgg_model.predict(img_data)
    return features

def is_shoe_image(img_path, binary_classifier_path='models/binary_shoe_classifier.pkl', scaler_path='models/scaler.pkl'):
    # Load the binary classifier and the scaler
    if not os.path.exists(binary_classifier_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Model or scaler not found at {binary_classifier_path} or {scaler_path}. Please train the classifier.")

    binary_classifier = joblib.load(binary_classifier_path)
    scaler = joblib.load(scaler_path)

    # Extract features from the image
    features = extract_features(img_path)
    
    # Scale the features
    features_scaled = scaler.transform(features)
    
    # Predict using the binary classifier
    prediction = binary_classifier.predict(features_scaled)
    
    # Return True if it's a shoe, False otherwise
    return prediction[0] == 1
