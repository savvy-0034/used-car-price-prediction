import pandas as pd
import joblib

# Load trained model
model = joblib.load("linear_regression_model.pkl")

# Load training data to get the exact feature columns
X_train = pd.read_csv("X_train.csv")

print("Used Car Price Prediction System")
print("----------------------------------")

# Get user input
car_name = input("Enter Car Name: ")
year = int(input("Enter Year: "))
present_price = float(input("Enter Present Price: "))
kms_driven = int(input("Enter Kms Driven: "))

fuel_type = input("Enter Fuel Type (Petrol/Diesel/CNG): ").strip()
seller_type = input("Enter Seller Type (Dealer/Individual): ").strip()
transmission = input("Enter Transmission (Manual/Automatic): ").strip()
owner = int(input("Enter Number of Previous Owners: "))

# Create input dataframe
input_data = pd.DataFrame(0, index=[0], columns=X_train.columns)

# Numerical features
input_data["Year"] = year
input_data["Present_Price"] = present_price
input_data["Kms_Driven"] = kms_driven
input_data["Owner"] = owner

# Car name
car_column = "Car_Name_" + car_name

if car_column in input_data.columns:
    input_data[car_column] = 1
else:
    print("\nWarning: Car name not found in training data.")
    print("Prediction cannot be made for this car name.")
    exit()

# Fuel type
fuel_column = "Fuel_Type_" + fuel_type

if fuel_column in input_data.columns:
    input_data[fuel_column] = 1
else:
    print("\nInvalid Fuel Type.")
    exit()

# Seller type
if seller_type == "Individual":
    input_data["Seller_Type_Individual"] = 1
elif seller_type != "Dealer":
    print("\nInvalid Seller Type.")
    exit()

# Transmission
if transmission == "Manual":
    input_data["Transmission_Manual"] = 1
elif transmission != "Automatic":
    print("\nInvalid Transmission.")
    exit()

# Make prediction
prediction = model.predict(input_data)

print("\n----------------------------------")
print("Predicted Selling Price:", round(prediction[0], 2), "Lakhs")
print("----------------------------------")