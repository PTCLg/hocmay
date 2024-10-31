# Bài tập 12: Dự đoán số lượt đặt phòng khách sạn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# Step 1: Load the dataset
data = pd.read_csv('dataset/hotel_bookings.csv')

# Step 2: Preprocessing and feature selection
# Remove irrelevant or target columns, assuming 'is_canceled' is the target variable
X = data.drop(['is_canceled', 'reservation_status'], axis=1)
y = data['is_canceled']

# List of categorical columns to encode
categorical_cols = X.select_dtypes(include=['object']).columns

# Define a preprocessor to handle categorical variables
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'  # Keep the numerical features as is
)

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Define a pipeline for Linear Regression model
lin_reg_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('regressor', LinearRegression())])

# Step 5: Train the Linear Regression model
lin_reg_pipeline.fit(X_train, y_train)

# Step 6: Predict using the Linear Regression model
y_pred_lr = lin_reg_pipeline.predict(X_test)

# Step 7: Evaluate the Linear Regression model
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("Linear Regression Results:")
print(f"MAE: {mae_lr}")
print(f"MSE: {mse_lr}")
print(f"R-squared: {r2_lr}")

# Step 8: Define a pipeline for Random Forest Regression model
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])

# Step 9: Train the Random Forest model
rf_pipeline.fit(X_train, y_train)

# Step 10: Predict using the Random Forest model
y_pred_rf = rf_pipeline.predict(X_test)

# Step 11: Evaluate the Random Forest model
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("\nRandom Forest Regression Results:")
print(f"MAE: {mae_rf}")
print(f"MSE: {mse_rf}")
print(f"R-squared: {r2_rf}")

# Step 12: Conclusion - Compare models
print("\nModel Comparison:")
if r2_rf > r2_lr:
    print("Random Forest performs better with a higher R-squared.")
else:
    print("Linear Regression performs better with a higher R-squared.")
