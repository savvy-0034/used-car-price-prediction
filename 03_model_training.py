import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load preprocessed training and testing datasets

X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").squeeze()
y_test = pd.read_csv("y_test.csv").squeeze()

print("Preprocessed datasets loaded successfully!")

print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)

# Create Linear Regression model

model = LinearRegression()

# Train the model

model.fit(X_train, y_train)

print("\nLinear Regression model trained successfully!")
import joblib

joblib.dump(model, "linear_regression_model.pkl")
print("\nModel saved successfully!")
# Make predictions

y_pred = model.predict(X_test)

print("\nFirst 5 predictions:")
print(y_pred[:5])

# Evaluate the model

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R2 Score:", r2)