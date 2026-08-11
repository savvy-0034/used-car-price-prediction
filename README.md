# Used Car Price Prediction 🚗💰

## Project Overview

This project aims to build a robust Machine Learning pipeline to predict used car prices. To achieve a comprehensive analysis, our team divided the project into two main pipelines:

1. **Structured Data Analysis**: Predicting prices based on numerical and categorical features (e.g., Year, Mileage, Fuel Type) using the Cardekho dataset.
2. **Unstructured Data Analysis (NLP)**: Predicting prices based solely on user-generated vehicle descriptions using the Craigslist dataset.

## Datasets Used

- **Cardekho Dataset (Indian Market):** Used for structured data modeling (Columns: Car_Name, Year, Selling_Price, Kms_Driven, Fuel_Type, etc.).
- **Craigslist Vehicles Dataset (US Market):** Used for Natural Language Processing (NLP). A sample of 10,000 records was utilized to optimize computational performance.

## Team Contributions & Methodology

### 1. Structured Data Pipeline (Members 1 - 4)

- **Data Cleaning & EDA:** Handled missing values, removed outliers, and analyzed feature correlations.
- **Preprocessing:** Encoded categorical variables (Fuel_Type, Seller_Type, Transmission) and scaled numerical features.
- **Model Training:** Trained multiple regression models (e.g., Linear Regression, Decision Trees) to predict `Selling_Price`.

### 2. NLP & Text Processing Pipeline (Member 5 - ITBIN-2313-0014)

- **Text Preprocessing:** Converted vehicle descriptions to lowercase and removed special characters using regular expressions.
- **Feature Extraction:** Applied `TfidfVectorizer` (max_features=500, English stop words removed) to convert text into numerical format.
- **Hyperparameter Tuning:** Utilized `GridSearchCV` to find the optimal parameters for the Random Forest Regressor (`max_depth=20`, `n_estimators=100`).
- **Cross-Validation:** Ensured model stability using 5-Fold Cross-Validation.

## Model Evaluation & Results

### Structured Data Models (Cardekho)

| Model                     | MAE                  | RMSE                 | R² Score             |
| :------------------------ | :------------------- | :------------------- | :------------------- |
| **Best Structured Model** | [Member 1-4 to fill] | [Member 1-4 to fill] | [Member 1-4 to fill] |

### NLP Models (Craigslist)

| Model                           | MAE           | RMSE          | R² Score   |
| :------------------------------ | :------------ | :------------ | :--------- |
| **Linear Regression (Untuned)** | $5,873.28     | $7,740.03     | 0.5817     |
| **Random Forest (Tuned)**       | **$5,214.45** | **$7,259.57** | **0.6320** |

_Note: The NLP Random Forest model achieved an Average K-Fold R² Score of 0.6477, proving that textual descriptions alone hold significant predictive power._

## Known Limitations (Data-Matching)

As initially documented in our proposal, the structured data (Cardekho) and textual data (Craigslist) belong to two different regional markets. Merging them row-by-row would create a data-matching problem. Therefore, the NLP analysis and structured data analysis were kept strictly isolated to evaluate their respective predictive powers independently, without fabricating connections between unrelated datasets.

## How to Run

1. Install dependencies: `pip install pandas numpy scikit-learn`
2. Run structured models: `python 03_model_training.py`
3. Run NLP models: Open and execute `05_nlp_evaluation.ipynb`
