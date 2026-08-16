import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
y_test = pd.read_csv("y_test.csv").values.ravel()

print("Training Models as per Project Proposal...\n")

# ==========================================
# 2. Linear Regression Model
# ==========================================
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

print("--- 1. Linear Regression Results ---")
print(f"MAE: {mean_absolute_error(y_test, lr_preds):.4f}")
print(f"MSE: {mean_squared_error(y_test, lr_preds):.4f}")
print(f"R2 Score: {r2_score(y_test, lr_preds):.4f}\n")

# ==========================================
# 3. Artificial Neural Network (ANN) Model
# ==========================================
ann_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
ann_model.fit(X_train, y_train)
ann_preds = ann_model.predict(X_test)

print("--- 2. ANN Results ---")
print(f"MAE: {mean_absolute_error(y_test, ann_preds):.4f}")
print(f"MSE: {mean_squared_error(y_test, ann_preds):.4f}")
print(f"R2 Score: {r2_score(y_test, ann_preds):.4f}\n")

# ==========================================
# 4. Random Forest Model (Best Model)
# ==========================================
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

print("--- 3. Random Forest Results ---")
print(f"MAE: {mean_absolute_error(y_test, rf_preds):.4f}")
print(f"MSE: {mean_squared_error(y_test, rf_preds):.4f}")
print(f"R2 Score: {r2_score(y_test, rf_preds):.4f}\n")


# ==========================================

joblib.dump(rf_model, "random_forest_model.pkl")
print("✅ Best Model (Random Forest) saved successfully for the Web App!")