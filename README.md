# Used Car Price Prediction 🚗💰

**IT41033 NIA Mini Project - Automated Used Car Price Prediction System**

## Project Overview

This project aims to build a robust Machine Learning pipeline to predict used car prices. To achieve a comprehensive analysis, our team divided the project into two main pipelines:

1. **Structured Data Analysis:** Predicting prices based on numerical and categorical features such as Year, Mileage, Fuel Type, Present Price, and other vehicle attributes using the Cardekho dataset.
2. **Unstructured Data Analysis (NLP):** Predicting prices based on user-generated vehicle descriptions using the Craigslist dataset.

## Datasets Used

- **Cardekho Dataset (Indian Market):** Used for structured data modeling. It contains features such as `Car_Name`, `Year`, `Selling_Price`, `Present_Price`, `Kms_Driven`, `Fuel_Type`, `Seller_Type`, `Transmission`, and `Owner`.
- **Craigslist Vehicles Dataset (US Market):** Used for Natural Language Processing (NLP). A sample of 10,000 records was utilized to optimize computational performance.

---

## Team Contributions & Methodology (Work Division)

### Member 1 – Data Collection & Understanding

**Student ID:** [To be updated by Member 1]

- **Datasets:** Integrated and documented the Cardekho and Craigslist datasets.
- **Dataset Documentation:** Created the data dictionary and outlined feature descriptions.
- **Initial Data Analysis:** Performed the preliminary examination of data types, structure, and basic statistics.

### Member 2 – Data Preprocessing

**Student ID:** [To be updated by Member 2]

- **Data Cleaning:** Handled missing values and removed duplicate records.
- **Outlier Handling:** Addressed outliers using the Interquartile Range (IQR) method.
- **Encoding & Transformation:** Applied Categorical Encoding (One-hot encoding), Binning for continuous variables, and Log transformations to normalize skewed data.

### Member 3 – EDA, Feature Engineering & PCA

**Student ID:** ITBIN-2313-0049

**Exploratory Data Analysis (EDA):**
The Cardekho dataset was analyzed to understand its structure, quality, distributions, and relationships between features.

- Original dataset contained 301 records and 9 features. After removing 2 duplicates, 299 unique records remained.
- Examined missing values, dataset dimensions, and data types.

**Feature Engineering:**

- **Car Age:** Created a `Car_Age` feature (`Car_Age = 2026 - Year`) to represent the age of each vehicle.
- **Categorical Encoding:** Applied one-hot encoding for `Fuel_Type_Diesel`, `Fuel_Type_Petrol`, `Seller_Type_Individual`, and `Transmission_Manual`.
- **Price Ratio:** Created a `Price_Ratio` feature (`Selling_Price / Present_Price`) for exploratory analysis (excluded from ML models to prevent target leakage).

**Correlation Analysis:**

- Strong positive relationship between `Present_Price` and `Selling_Price` (Correlation = 0.876).
- Weak negative relationship between `Car_Age` and `Selling_Price` (Correlation = -0.234).

**Principal Component Analysis (PCA):**
Applied PCA to reduce dimensionality while preserving dataset information.

- Features were standardized using `StandardScaler`.
- **8 original features** were reduced to **6 principal components**, preserving **95.09%** of the total variance.

### Member 4 – Machine Learning

**Student ID:** [To be updated by Member 4]

- **Model Selection:** Implemented various algorithms including Linear Regression, Random Forest, and Artificial Neural Networks (ANN).
- **Model Training:** Trained the models on the preprocessed and PCA-transformed Cardekho dataset.
- **Evaluation:** Compared models to find the best performing algorithm for structured data.

### Member 5 – NLP & Evaluation

**Student ID:** ITBIN-2313-0014

- **Text Preprocessing:** Converted Craigslist vehicle descriptions to lowercase and removed special characters using regular expressions.
- **Feature Extraction:** Applied `TfidfVectorizer` (`max_features=500`) with English stop-word removal.
- **Hyperparameter Tuning:** Utilized `GridSearchCV` to find optimal parameters for the Random Forest Regressor.
- **Cross-Validation:** Used 5-Fold Cross-Validation to evaluate model stability.
- **Final Model Comparison:** Evaluated the NLP models using MAE, RMSE, and R² metrics to finalize the text-based prediction pipeline.

---

## Generated Files

The following files were generated as part of the pipeline:

| File                                   | Description                                  |
| :------------------------------------- | :------------------------------------------- |
| `03_eda_feature_engineering_pca.ipynb` | EDA, feature engineering, and PCA notebook   |
| `05_nlp_evaluation.ipynb`              | NLP text processing and evaluation notebook  |
| `cleaned_eda_data.csv`                 | Cleaned dataset after duplicate removal      |
| `pca_features.csv`                     | Dataset containing the selected PCA features |
| `pca_model.pkl`                        | Saved PCA transformation                     |
| `pca_scaler.pkl`                       | Saved StandardScaler transformation          |

---

## Final Model Evaluation & Results

### 1. Structured Data Models (Cardekho) - By Member 4

| Model                 | MAE                | RMSE               | R² Score           |
| :-------------------- | :----------------- | :----------------- | :----------------- |
| **Linear Regression** | [Member 4 to fill] | [Member 4 to fill] | [Member 4 to fill] |
| **Random Forest**     | [Member 4 to fill] | [Member 4 to fill] | [Member 4 to fill] |
| **ANN**               | [Member 4 to fill] | [Member 4 to fill] | [Member 4 to fill] |

### 2. NLP Models (Craigslist) - By Member 5

| Model                           | MAE           | RMSE          | R² Score   |
| :------------------------------ | :------------ | :------------ | :--------- |
| **Linear Regression (Untuned)** | $5,873.28     | $7,740.03     | 0.5817     |
| **Random Forest (Tuned)**       | **$5,214.45** | **$7,259.57** | **0.6320** |

_Note: The NLP Random Forest model achieved an Average K-Fold R² Score of 0.6477, proving that textual descriptions alone hold significant predictive power._

---

## Known Limitations

As initially documented, the structured data (Cardekho - Indian Market) and textual data (Craigslist - US Market) belong to different regional markets. Therefore, the datasets were not merged row-by-row. The structured and NLP pipelines were kept strictly independent to avoid creating artificial or unreliable relationships between unrelated vehicle records.

---

## How to Run

1. **Install Dependencies:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter streamlit
   ```

2. **Run the Web Application:**
   ```bash
   streamlit run app.py
