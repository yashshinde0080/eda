# eda

This Python script performs an end-to-end data analysis and prediction process on an e-commerce dataset using machine learning techniques.

# Key Steps:
1. Load & Clean Data:
   - Reads data from `ecommerce_data.csv`.
   - Converts columns to appropriate data types.
   - Fills missing values with column means (excluding `CustomerID`).

2. Exploratory Data Analysis (EDA):
   - Generates a pair plot and a heatmap to visualize correlations.
   
3. Feature Selection & Train-Test Split:
   - Selects key features (`Customer_Age`, `Customer_Satisfaction`, `Number_of_Purchases`) to predict `Revenue`.
   - Splits data into training and testing sets.

4. Model Training & Evaluation:
   - Compares `RandomForestRegressor` and `XGBoostRegressor` using cross-validation.
   - Chooses the best model based on R² score.
   - Performs hyperparameter tuning using `GridSearchCV` if XGBoost is the best model.
   
5. Final Evaluation:
   - Computes Mean Absolute Error (MAE) and R² score for the best model.
   - Generates a stack plot comparing actual vs predicted revenue.

6. Data Visualization:
   - Stem plots for customer age vs revenue and satisfaction.
   - Stack plots to visualize relationships between variables.
   - Pie chart for revenue distribution.
   - Bar plots for customer satisfaction and number of purchases.

This script effectively cleans data, trains machine learning models, evaluates performance, and visualizes important insights for decision-making in e-commerce analytics.
