# Used Vehicle Price Prediction 🚗💰

**IT41033 NIA Mini Project - Automated Used Vehicle Price Prediction System**

## Project Overview

This project aims to build a robust Machine Learning pipeline to predict used vehicle prices. To achieve a comprehensive analysis, our team divided the project into two main pipelines:

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

- Features were standardized using `StandardScaler`.
- **8 original features** were reduced to **6 principal components**, preserving **95.09%** of the total variance.

### Member 4 – Visualization & Web Deployment

**Student ID:** ITBIN-2313-0034

- **Visualization:** Generated Actual vs Predicted Vehicle Price visualization (`actual_vs_predicted.png`).
- **Deployment:** Built the interactive Streamlit-based web application interface (`app.py`) to deploy the final predictive model.

### Member 5 – Structured Model Training, NLP & Evaluation

**Student ID:** ITBIN-2313-0014 (Pulasthi Avinash)

- **Structured Data Modeling:** Developed, trained, and evaluated multiple regression models (Linear Regression, Artificial Neural Network, and Random Forest Regressor) for the Cardekho dataset.
- **Model Export:** Exported the highest-performing structured model (`random_forest_model.pkl`) for web deployment.
- **Text Preprocessing & NLP:** Converted Craigslist vehicle descriptions to lowercase and removed special characters using regular expressions.
- **NLP Feature Extraction:** Applied `TfidfVectorizer` (`max_features=500`) with English stop-word removal.
- **Hyperparameter Tuning:** Utilized `GridSearchCV` to find optimal parameters for the NLP Random Forest Regressor.
- **Final Model Comparison:** Evaluated both structured and NLP models using MAE, MSE, RMSE, and R² metrics to finalize the pipelines.

---

## Generated Files

The following files are generated as part of the pipeline:

| File                                   | Description                                             |
| :------------------------------------- | :------------------------------------------------------ |
| `cleaned_eda_data.csv`                 | Cleaned dataset after duplicate removal                 |
| `X_train.csv` / `X_test.csv`           | Training and testing input features                     |
| `y_train.csv` / `y_test.csv`           | Training and testing target variables                   |
| `03_eda_feature_engineering_pca.ipynb` | EDA, feature engineering, and PCA notebook              |
| `05_nlp_evaluation.ipynb`              | NLP text processing and evaluation notebook             |
| `pca_features.csv`                     | Dataset containing the selected PCA features            |
| `pca_model.pkl`                        | Saved PCA transformation                                |
| `pca_scaler.pkl`                       | Saved StandardScaler transformation                     |
| `random_forest_training.py`            | Model training script for structured data (LR, ANN, RF) |
| `random_forest_model.pkl`              | Saved Random Forest model for web app deployment        |
| `app.py`                               | Streamlit web application script                        |
| `actual_vs_predicted.png`              | Visualization of the model's accuracy                   |

---

## Final Model Evaluation & Results

### 1. Structured Data Models (Cardekho) - Trained by Member 5

| Model                   | MAE     | MSE       | R² Score    |
| :---------------------- | :------ | :-------- | :---------- |
| **Linear Regression**   | 0.1352  | 0.0395    | 0.9208      |
| **ANN (MLP Regressor)** | 48.4459 | 5410.0599 | -10857.1916 |
| **Random Forest**       | 0.0927  | 0.0213    | **0.9572**  |

### 2. NLP Models (Craigslist) - Trained by Member 5

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
2. **Run the Preprocessing and Training Pipeline:**
   ```bash
   python 02_data_preprocessing.py
   python random_forest_training.py
   ```
3. **Launch the Web Application:**
   ```bash
   streamlit run app.py
   ```  