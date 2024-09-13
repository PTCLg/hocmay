# train_binary_classifier.py

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def load_features_and_labels():
    # Load pre-extracted features for shoe and non-shoe images
    # X should be the feature vectors (e.g., from a CNN like VGG16), and y should be the labels (1 for shoe, 0 for non-shoe)
    # Placeholder implementation - replace with actual data loading
    X = np.random.rand(1000, 512)  # 1000 samples, 512 features (example)
    y = np.random.randint(0, 2, 1000)  # Random binary labels for now (replace with actual labels)
    return X, y

# Load the features and labels
X, y = load_features_and_labels()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train an SVM classifier
binary_classifier = SVC(kernel='linear')
binary_classifier.fit(X_train, y_train)

# Evaluate the model
y_pred = binary_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Binary classifier accuracy: {accuracy * 100:.2f}%")

# Save the model and the scaler
joblib.dump(binary_classifier, 'models/binary_shoe_classifier.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
