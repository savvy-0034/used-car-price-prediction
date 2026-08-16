import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load the cleaned dataset
df = pd.read_csv("cleaned_car_data.csv")

print("Cleaned dataset loaded successfully!")
print("\nOriginal dataset shape:", df.shape)

# ==========================================
# NEW ADDITIONS: Member 2 Preprocessing Steps
# ==========================================

# 1. IQR Outlier Removal (Present_Price)
Q1 = df['Present_Price'].quantile(0.25)
Q3 = df['Present_Price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out the outliers
df = df[(df['Present_Price'] >= lower_bound) & (df['Present_Price'] <= upper_bound)]
print("\nDataset shape after IQR outlier removal:", df.shape)

# 2. Binning (Discretization) for Kms_Driven
bins = [0, 30000, 60000, 100000, np.inf]
labels = ['Low', 'Medium', 'High', 'Very_High']
df['Mileage_Category'] = pd.cut(df['Kms_Driven'], bins=bins, labels=labels)

# 3. Log Transformation for Selling_Price
df['Selling_Price'] = np.log1p(df['Selling_Price'])

# ==========================================
# CATEGORICAL ENCODING & SPLITTING
# ==========================================

# Convert categorical columns into numerical values (Included new 'Mileage_Category')
df = pd.get_dummies(
    df,
    columns=["Car_Name", "Fuel_Type", "Seller_Type", "Transmission", "Mileage_Category"],
    drop_first=True,
    dtype=int
)

print("\nDataset after categorical encoding:")
print(df.head())
print("\nNew dataset shape:", df.shape)

# Separate features and target variable
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

print("\nFeatures (X) shape:", X.shape)
print("Target (y) shape:", y.shape)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Save preprocessed datasets
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("\nPreprocessed datasets saved successfully!")