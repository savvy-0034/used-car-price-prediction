# used-car-price-prediction

**IT41033 NIA Mini Project - Automated Used Car Price Prediction System**

# Used Car Price Prediction 🚗💰

## Project Overview

This project aims to build a robust Machine Learning pipeline to predict used car prices. To achieve a comprehensive analysis, our team divided the project into two main pipelines:

1. **Structured Data Analysis:** Predicting prices based on numerical and categorical features such as Year, Mileage, Fuel Type, Present Price, and other vehicle attributes using the Cardekho dataset.
2. **Unstructured Data Analysis (NLP):** Predicting prices based on user-generated vehicle descriptions using the Craigslist dataset.

## Datasets Used

- **Cardekho Dataset (Indian Market):** Used for structured data modeling. It contains features such as `Car_Name`, `Year`, `Selling_Price`, `Present_Price`, `Kms_Driven`, `Fuel_Type`, `Seller_Type`, `Transmission`, and `Owner`.
- **Craigslist Vehicles Dataset (US Market):** Used for Natural Language Processing (NLP). A sample of 10,000 records was utilized to optimize computational performance.

---

# Team Contributions & Methodology

## 1. Structured Data Pipeline (Members 1 - 4)

The structured data pipeline includes data cleaning, exploratory analysis, feature engineering, preprocessing, model training, and evaluation.

### Member 3 - EDA, Feature Engineering & PCA

**Student ID:** ITBIN-2313-0049

The responsibilities of Member 3 include Exploratory Data Analysis (EDA), Feature Engineering, and Principal Component Analysis (PCA).

### Exploratory Data Analysis (EDA)

The Cardekho dataset was analyzed to understand its structure, quality, distributions, and relationships between features.

Key EDA activities performed:

- Examined the dataset dimensions and data types.
- Checked for missing values.
- Identified and removed duplicate records.
- Analyzed numerical feature correlations.
- Investigated the relationship between vehicle age and selling price.
- Analyzed the relationship between present price and selling price.
- Examined categorical features such as fuel type, seller type, and transmission.

### Data Cleaning

The original dataset contained:

- **301 records**
- **9 features**
- **0 missing values**
- **2 duplicate records**

The two duplicate records were removed, resulting in:

- **299 unique records**

### Feature Engineering

Several new features and transformations were introduced to improve the representation of the dataset.

#### Car Age

A `Car_Age` feature was created from the `Year` attribute:

`Car_Age = 2026 - Year`

This represents the age of each vehicle and provides a more meaningful feature for price prediction.

#### Categorical Encoding

Categorical variables were converted into numerical representations using one-hot encoding.

The following features were created:

- `Fuel_Type_Diesel`
- `Fuel_Type_Petrol`
- `Seller_Type_Individual`
- `Transmission_Manual`

#### Price Ratio

A `Price_Ratio` feature was created for exploratory analysis:

`Price_Ratio = Selling_Price / Present_Price`

However, this feature was excluded from PCA and machine-learning features because it directly uses the target variable `Selling_Price`, which could introduce target leakage.

### Correlation Analysis

A strong positive relationship was identified between `Present_Price` and `Selling_Price`:

- **Correlation = 0.876**

A weak negative relationship was identified between `Car_Age` and `Selling_Price`:

- **Correlation = -0.234**

These findings helped identify important factors related to used car prices.

---

## 2. Principal Component Analysis (PCA)

PCA was applied to reduce the dimensionality of the engineered feature set while preserving most of the information in the dataset.

### PCA Features

The following 8 features were used as PCA inputs:

- `Present_Price`
- `Kms_Driven`
- `Owner`
- `Car_Age`
- `Fuel_Type_Diesel`
- `Fuel_Type_Petrol`
- `Seller_Type_Individual`
- `Transmission_Manual`

Before applying PCA, the features were standardized using `StandardScaler`.

### PCA Results

The cumulative explained variance was:

| Principal Components | Cumulative Variance |
| :------------------- | :------------------ |
| PC1                   | 33.94%              |
| PC1-PC2               | 54.00%              |
| PC1-PC3               | 67.41%              |
| PC1-PC4               | 79.45%              |
| PC1-PC5               | 89.62%              |
| **PC1-PC6**           | **95.09%**          |
| PC1-PC7               | 99.74%              |
| PC1-PC8               | 100.00%             |

Based on the explained variance, **6 principal components were retained**, preserving **95.09% of the total variance**.

Therefore:

**8 original features → 6 principal components**

This dimensionality reduction provides a more compact feature representation while retaining most of the information in the original feature set.

---

## 3. NLP & Text Processing Pipeline (Member 5 - ITBIN-2313-0014)

- **Text Preprocessing:** Converted vehicle descriptions to lowercase and removed special characters using regular expressions.
- **Feature Extraction:** Applied `TfidfVectorizer` with `max_features=500` and English stop-word removal.
- **Hyperparameter Tuning:** Utilized `GridSearchCV` to find optimal parameters for the Random Forest Regressor.
- **Cross-Validation:** Used 5-Fold Cross-Validation to evaluate model stability.

---

# Generated Files

The following files were generated as part of the EDA, Feature Engineering, and PCA process:

| File | Description |
| :--- | :--- |
| `03_eda_feature_engineering_pca.ipynb` | EDA, feature engineering, and PCA notebook |
| `cleaned_eda_data.csv` | Cleaned dataset after duplicate removal |
| `pca_features.csv` | Dataset containing the selected PCA features |
| `pca_model.pkl` | Saved PCA transformation |
| `pca_scaler.pkl` | Saved StandardScaler transformation |

---

# Model Evaluation & Results

## Structured Data Models (Cardekho)

| Model | MAE | RMSE | R² Score |
| :--- | :--- | :--- | :--- |
| **Best Structured Model** | [Member 1-4 to fill] | [Member 1-4 to fill] | [Member 1-4 to fill] |

## NLP Models (Craigslist)

| Model | MAE | RMSE | R² Score |
| :--- | :--- | :--- | :--- |
| **Linear Regression (Untuned)** | $5,873.28 | $7,740.03 | 0.5817 |
| **Random Forest (Tuned)** | **$5,214.45** | **$7,259.57** | **0.6320** |

The NLP Random Forest model achieved an Average K-Fold R² Score of 0.6477.

---

# Known Limitations

The structured data (Cardekho) and textual data (Craigslist) belong to different regional markets.

Therefore, the two datasets were not merged row-by-row. The structured and NLP pipelines were kept independent to avoid creating artificial or unreliable relationships between unrelated vehicle records.

---

# How to Run

### Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter