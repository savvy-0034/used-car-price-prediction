import pandas as pd

# Load the dataset
df = pd.read_csv("car data.csv")

# Display basic information
print("Dataset loaded successfully!")
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())
# Check for duplicate rows
duplicate_count = df.duplicated().sum()

print("\nDuplicate rows:")
print(duplicate_count)
# Remove duplicate rows
df = df.drop_duplicates()

print("\nDataset shape after removing duplicates:")
print(df.shape)

print("\nDuplicate rows after cleaning:")
print(df.duplicated().sum())
# Check numerical columns for invalid negative values

print("\nNegative values check:")

print("Selling_Price:", (df["Selling_Price"] < 0).sum())
print("Present_Price:", (df["Present_Price"] < 0).sum())
print("Kms_Driven:", (df["Kms_Driven"] < 0).sum())
print("Owner:", (df["Owner"] < 0).sum())

# Check Year range
print("\nYear range:")
print("Minimum Year:", df["Year"].min())
print("Maximum Year:", df["Year"].max())

# Check unique categorical values
print("\nFuel Types:")
print(df["Fuel_Type"].unique())

print("\nSeller Types:")
print(df["Seller_Type"].unique())

print("\nTransmission Types:")
print(df["Transmission"].unique())
# Save cleaned dataset
df.to_csv("cleaned_car_data.csv", index=False)

print("\nCleaned dataset saved successfully!")