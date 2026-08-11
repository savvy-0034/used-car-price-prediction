import pandas as pd

# Load the cleaned dataset
df = pd.read_csv("cleaned_car_data.csv")

print("Cleaned dataset loaded successfully!")

print("\nDataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

# Check categorical columns
print("\nCategorical columns:")
print(df[["Car_Name", "Fuel_Type", "Seller_Type", "Transmission"]].head())

print("\nUnique values:")
print("Fuel Type:", df["Fuel_Type"].unique())
print("Seller Type:", df["Seller_Type"].unique())
print("Transmission:", df["Transmission"].unique())
# Convert categorical columns into numerical values

df = pd.get_dummies(
    df,
    columns=["Car_Name", "Fuel_Type", "Seller_Type", "Transmission"],
    drop_first=True,
    dtype=int
)

print("\nDataset after categorical encoding:")
print(df.head())

print("\nNew dataset shape:")
print(df.shape)

print("\nNew column names:")
print(df.columns.tolist())
# Separate features and target variable

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

print("\nFeatures (X) shape:")
print(X.shape)

print("\nTarget (y) shape:")
print(y.shape)

print("\nTarget variable:")
print(y.head())
# Split dataset into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)

print("\nTraining target shape:")
print(y_train.shape)
print("\nTesting target shape:")
print(y_test.shape)
# Save preprocessed datasets

X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("\nPreprocessed datasets saved successfully!")