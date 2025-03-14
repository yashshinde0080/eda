import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load Dataset
df = pd.read_csv('ecommerce_data.csv')

# Display the first 20 rows in the terminal
print("First 20 rows of the dataset:")
print(df.head(20).to_string(index=False))

# Clean the data
# Convert columns to appropriate data types if necessary
df['Revenue'] = df['Revenue'].astype(float)
df['Customer_Age'] = df['Customer_Age'].astype(int)
df['Customer_Satisfaction'] = df['Customer_Satisfaction'].astype(float)

# Display first few rows in the desired format
print("E-commerce Data:")
print(df.head().to_string(index=False))

# Check for missing values and fill them (excluding the 'CustomerID' column)
df.fillna(df.drop(columns=['CustomerID']).mean(), inplace=True)

# Exploratory Data Analysis (EDA)
# Pairplot
sns.pairplot(df.drop(columns=['CustomerID']), diag_kind='kde')
plt.suptitle('Pairplot of E-commerce Data', y=1.02)
plt.show()

# Correlation Matrix (excluding the 'CustomerID' column)
plt.figure(figsize=(8, 6))
sns.heatmap(df.drop(columns=['CustomerID']).corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation')
plt.show()

# Feature Selection
X = df[['Customer_Age', 'Customer_Satisfaction', 'Number_of_Purchases']]
y = df['Revenue']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection & Training
models = {
    'RandomForest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42)
}

best_model = None
best_score = -np.inf

for name, model in models.items():
    # Cross-validation for better evaluation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    print(f'{name} Cross-Validation R² Scores: {cv_scores}')
    print(f'{name} Average Cross-Validation R² Score: {np.mean(cv_scores):.2f}')
    
    # Fit the model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    print(f'{name} Test R² Score: {score:.2f}')
    
    if score > best_score:
        best_score = score
        best_model = model

# Hyperparameter Tuning for XGBoost (if it's the best model)
if isinstance(best_model, XGBRegressor):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    grid_search = GridSearchCV(best_model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print(f'Best Parameters for XGBoost: {grid_search.best_params_}')

# Final Evaluation
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Best Model: {best_model}')
print(f'MAE: {mae:.2f}')
print(f'R² Score: {r2:.2f}')

# Stack Plot of Predictions
plt.figure(figsize=(10, 6))
plt.stackplot(range(len(y_test)), y_test, y_pred, labels=['Actual Revenue', 'Predicted Revenue'], alpha=0.6)
plt.xlabel("Customers")
plt.ylabel("Revenue")
plt.title("Actual vs Predicted Revenue")
plt.legend(loc='upper left')
plt.show()

# Additional Visualizations
# Stem Plot for Customer Age vs Customer Satisfaction
plt.figure(figsize=(10, 6))
plt.stem(df['Customer_Age'], df['Customer_Satisfaction'], linefmt='b-', markerfmt='bo', basefmt='r-')
plt.xlabel('Customer Age')
plt.ylabel('Customer Satisfaction')
plt.title('Customer Age vs Customer Satisfaction')
plt.grid(True)
plt.show()

# Stem Plot for Customer Age vs Revenue
plt.figure(figsize=(10, 6))
plt.stem(df['Customer_Age'], df['Revenue'], linefmt='b-', markerfmt='bo', basefmt='r-')
plt.xlabel('Customer Age')
plt.ylabel('Revenue')
plt.title('Customer Age vs Revenue')
plt.grid(True)
plt.show()

# Stack Plot for Customer Age, Customer Satisfaction, and Revenue
plt.figure(figsize=(10, 6))
plt.stackplot(df.index, df['Customer_Age'], df['Customer_Satisfaction'], df['Revenue'], labels=['Customer Age', 'Customer Satisfaction', 'Revenue'], alpha=0.6)
plt.xlabel("Customer Index")
plt.ylabel("Values")
plt.title("Customer Age, Customer Satisfaction, and Revenue")
plt.legend(loc='upper left')
plt.show()

# Pie Chart for Distribution of Revenue
revenue_bins = [0, 1000, 2000, 3000, 4000, 5000]
revenue_labels = ['Revenue <1000', 'Revenue 1000-2000', 'Revenue 2000-3000', 'Revenue 3000-4000', 'Revenue 4000-5000']
df['Revenue_Category'] = pd.cut(df['Revenue'], bins=revenue_bins, labels=revenue_labels)
revenue_distribution = df['Revenue_Category'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(revenue_distribution, labels=revenue_labels, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('Distribution of Revenue')
plt.show()

# Bar Plot for Number of Purchases vs Customer Satisfaction (Single Bar for Each Customer)
plt.figure(figsize=(10, 6))
sns.barplot(x=df.index, y='Customer_Satisfaction', data=df, ci=None, palette='viridis')
plt.xlabel('Customer Index')
plt.ylabel('Customer Satisfaction')
plt.title('Customer Index vs Customer Satisfaction')
plt.show()

# Bar Plot for Number of Purchases vs Revenue (Single Bar for Each Customer)
plt.figure(figsize=(10, 6))
sns.barplot(x=df.index, y='Number_of_Purchases', data=df, ci=None, palette='viridis')
plt.xlabel('Customer Index')
plt.ylabel('Number of Purchases')
plt.title('Customer Index vs Number of Purchases')
plt.show()
