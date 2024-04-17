#!/usr/bin/env python
# coding: utf-8

# # DATA SET DETAILS

# Data Set Size: 9568 observations
# 
# Duration: Collected over six years
# 
# Operating Condition: The power plant was set to work with a full load.

# # Variables (Features)

# temperature: Temperature in degrees Celsius.
# 
# exhaust_vacuum: Vacuum of the exhaust in cm Hg.
# 
# amb_pressure: Ambient pressure in millibar.
# 
# r_humidity: Relative humidity in percentage.

# # Target Variable

# energy_production: Net hourly electrical energy output in MW.

# # IMPORT DATASET

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")


# In[2]:


filepath="C:\\Users\\chakri\\Downloads\\energy_production.csv"
df=pd.read_csv(filepath,sep=';')


# # Exploratory Data Analysis (EDA)

# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.isnull().sum()


# In[7]:


df.shape


# In[8]:


df[df.duplicated()].shape


# In[9]:


df[df.duplicated()]


# In[10]:


df1 = df.drop_duplicates()


# In[11]:


df1[df1.duplicated()]


# In[12]:


plt.figure(figsize=(10, 8))
sns.histplot(data=df1, x='energy_production', bins=30, kde=True)
plt.title('Distribution of Energy Production')
plt.xlabel('Energy Production (MW)')
plt.ylabel('Frequency')
plt.show()


# In[13]:


sns.pairplot(data=df)


# In[14]:


plt.figure(figsize=(12, 8))
for i, column in enumerate(df1.columns[:-1]):
    plt.subplot(2, 2, i + 1)
    sns.histplot(df1[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


# In[15]:


plt.figure(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


# In[16]:


df.corr()


# In[17]:


import pandas as pd
from scipy import stats

# Assuming your data is stored in a DataFrame named 'df1'

# Define the column for which you want to remove outliers
column_to_remove_outliers = 'r_humidity'

# Function to remove outliers using a combination of IQR and Z-score methods
def remove_outliers_combined(df, column, threshold_iqr=1.5, threshold_zscore=3):
    # Calculate IQR
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    # Calculate Z-scores
    z_scores = stats.zscore(df[column])
    
    # Identify outliers based on IQR and Z-score thresholds
    outliers_iqr = (df[column] < (Q1 - threshold_iqr * IQR)) | (df[column] > (Q3 + threshold_iqr * IQR))
    outliers_zscore = (z_scores < -threshold_zscore) | (z_scores > threshold_zscore)
    
    # Remove outliers
    df_filtered = df[~(outliers_iqr | outliers_zscore)]
    return df_filtered

# Remove outliers from the 'r_humidity' column using combined method
df1 = remove_outliers_combined(df1, column_to_remove_outliers)

# After removing outliers, you can proceed with further analysis or visualization


# In[18]:


import pandas as pd
from scipy import stats

# Assuming your data is stored in a DataFrame named 'df1'

# Define the column for which you want to remove outliers
column_to_remove_outliers = 'amb_pressure'

# Function to remove outliers using a combination of IQR and Z-score methods
def remove_outliers_combined(df, column, threshold_iqr=1.5, threshold_zscore=3):
    # Calculate IQR
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    # Calculate Z-scores
    z_scores = stats.zscore(df[column])
    
    # Identify outliers based on IQR and Z-score thresholds
    outliers_iqr = (df[column] < (Q1 - threshold_iqr * IQR)) | (df[column] > (Q3 + threshold_iqr * IQR))
    outliers_zscore = (z_scores < -threshold_zscore) | (z_scores > threshold_zscore)
    
    # Remove outliers
    df_filtered = df[~(outliers_iqr | outliers_zscore)]
    return df_filtered

# Remove outliers from the 'r_humidity' column using combined method
df1 = remove_outliers_combined(df1, column_to_remove_outliers)

# After removing outliers, you can proceed with further analysis or visualization


# In[19]:


# Create box plots to check for outliers
plt.figure(figsize=(12, 6))

# Box plot for temperature
plt.subplot(2, 2, 1)
sns.boxplot(x=df1['temperature'], color='skyblue')
plt.title('Box Plot for Temperature')

# Box plot for exhaust vacuum
plt.subplot(2, 2, 2)
sns.boxplot(x=df1['exhaust_vacuum'], color='salmon')
plt.title('Box Plot for Exhaust Vacuum')

# Box plot for ambient pressure
plt.subplot(2, 2, 3)
sns.boxplot(x=df1['amb_pressure'], color='lightgreen')
plt.title('Box Plot for Ambient Pressure')

# Box plot for relative humidity
plt.subplot(2, 2, 4)
sns.boxplot(x=df1['r_humidity'], color='gold')
plt.title('Box Plot for Relative Humidity')

plt.tight_layout()
plt.show()


# # MODEL EVALUATION

# In[20]:


X = df1[['temperature', 'exhaust_vacuum', 'amb_pressure', 'r_humidity']]
y = df1['energy_production']


# In[21]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[22]:


# Initialize and fit the linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)


# In[23]:


# Initialize and fit the Lasso regression model
lasso_model = Lasso(alpha=0.1)  # You can adjust the alpha parameter
lasso_model.fit(X_train, y_train)


# In[24]:


# Initialize and fit the Ridge regression model
ridge_model = Ridge(alpha=1.0)  # You can adjust the alpha parameter
ridge_model.fit(X_train, y_train)


# In[25]:


# Make predictions on the testing set for each model
linear_pred = linear_model.predict(X_test)
lasso_pred = lasso_model.predict(X_test)
ridge_pred = ridge_model.predict(X_test)


# 

# In[26]:


# Model Evaluation
# Calculate Mean Squared Error and R-squared for each model
linear_mse = mean_squared_error(y_test, linear_pred)
linear_r2 = r2_score(y_test, linear_pred)


# In[27]:


lasso_mse = mean_squared_error(y_test, lasso_pred)
lasso_r2 = r2_score(y_test, lasso_pred)


# In[28]:


ridge_mse = mean_squared_error(y_test, ridge_pred)
ridge_r2 = r2_score(y_test, ridge_pred)


# In[29]:


dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)


# In[30]:


# Initialize and fit the Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


# In[31]:


# Initialize and fit the XGBoost Regressor model
xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)


# In[32]:


# Make predictions on the testing set for each model
dt_pred = dt_model.predict(X_test)
rf_pred = rf_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)


# In[33]:


print("\nModel Evaluation:")
print("Linear Regression - Mean Squared Error:", linear_mse, "R-squared:", linear_r2)
print("Lasso Regression - Mean Squared Error:", lasso_mse, "R-squared:", lasso_r2)
print("Ridge Regression - Mean Squared Error:", ridge_mse, "R-squared:", ridge_r2)


# In[34]:


# Model Evaluation
# Calculate Mean Squared Error and R-squared for each model
dt_mse = mean_squared_error(y_test, dt_pred)
dt_r2 = r2_score(y_test, dt_pred)

rf_mse = mean_squared_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

xgb_mse = mean_squared_error(y_test, xgb_pred)
xgb_r2 = r2_score(y_test, xgb_pred)

print("\nModel Evaluation:")
print("Decision Tree - Mean Squared Error:", dt_mse, "R-squared:", dt_r2)
print("Random Forest - Mean Squared Error:", rf_mse, "R-squared:", rf_r2)
print("XGBoost - Mean Squared Error:", xgb_mse, "R-squared:", xgb_r2)


# In[35]:


# Showing predicted values for Linear Regression
print("\nPredicted values for Linear Regression:")
linear_results = pd.DataFrame({'Actual': y_test, 'Predicted': linear_pred})
print(linear_results.head())


# In[36]:


# Showing predicted values for Lasso Regression
print("\nPredicted values for Lasso Regression:")
lasso_results = pd.DataFrame({'Actual': y_test, 'Predicted': lasso_pred})
print(lasso_results.head())


# In[37]:


# Showing predicted values for Ridge Regression
print("\nPredicted values for Ridge Regression:")
ridge_results = pd.DataFrame({'Actual': y_test, 'Predicted': ridge_pred})
print(ridge_results.head())


# In[38]:


# Showing predicted values for Decision Tree
print("\nPredicted values for Decision Tree:")
dt_results = pd.DataFrame({'Actual': y_test, 'Predicted': dt_pred})
print(dt_results.head())


# In[39]:


# Showing predicted values for Random Forest
print("\nPredicted values for Random Forest:")
rf_results = pd.DataFrame({'Actual': y_test, 'Predicted': rf_pred})
print(rf_results.head())


# In[40]:


# Showing predicted values for XGBoost
print("\nPredicted values for XGBoost:")
xgb_results = pd.DataFrame({'Actual': y_test, 'Predicted': xgb_pred})
print(xgb_results.head())


# In[41]:


# Create a bar chart to compare model performance
models = ['Linear Regression', 'Lasso Regression', 'Ridge Regression', 
          'Decision Tree', 'Random Forest', 'XGBoost']
mse_values = [linear_mse, lasso_mse, ridge_mse, dt_mse, rf_mse, xgb_mse]
r2_values = [linear_r2, lasso_r2, ridge_r2, dt_r2, rf_r2, xgb_r2]

plt.figure(figsize=(10, 6))

# Plot Mean Squared Error
plt.subplot(1, 2, 1)
plt.bar(models, mse_values, color='skyblue')
plt.title('Mean Squared Error Comparison')
plt.xlabel('Models')
plt.ylabel('Mean Squared Error')
plt.xticks(rotation=45)
for i in range(len(models)):
    plt.text(i, mse_values[i], round(mse_values[i], 2), ha='center', va='bottom')

# Plot R-squared
plt.subplot(1, 2, 2)
plt.bar(models, r2_values, color='lightgreen')
plt.title('R-squared Comparison')
plt.xlabel('Models')
plt.ylabel('R-squared')
plt.xticks(rotation=45)
for i in range(len(models)):
    plt.text(i, r2_values[i], round(r2_values[i], 3), ha='center', va='bottom')

plt.tight_layout()
plt.show()


# In[42]:


# Showing predicted values for XGBoost
print("\nPredicted values for XGBoost:")
xgb_results = pd.DataFrame({'Actual': y_test, 'Predicted': xgb_pred})
print(xgb_results.head())


# In[43]:


pip install Flask


# In[44]:


pip install streamlit


# In[50]:





# In[59]:


pip install streamlit


# In[ ]:





# In[70]:


import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Assuming X and y are your features and target variable
# X should contain the feature columns and y should contain the target variable column

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train your model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'trained_model.pkl')


# In[78]:


import streamlit as st
import joblib

# Load the trained model
model = joblib.load("trained_model.pkl")

# Define the Streamlit app
def main():
    # Set title and description
    st.title('Energy Production Prediction')
    st.write('Enter the input features to predict energy production.')

    # Input form for user to enter feature values
    temperature = st.number_input('Temperature (°C)')
    exhaust_vacuum = st.number_input('Exhaust Vacuum (cm Hg)')
    amb_pressure = st.number_input('Ambient Pressure (millibar)')
    r_humidity = st.number_input('Relative Humidity (%)')

    # Predict button
    if st.button('Predict'):
        try:
            # Make prediction using the loaded model
            prediction = model.predict([[temperature, exhaust_vacuum, amb_pressure, r_humidity]])
            st.write(f'Predicted Energy Production: {prediction[0]} MW')
        except Exception as e:
            st.error(f"Error predicting energy production: {e}")
            print(f"Error predicting energy production: {e}")  # Print error for debugging

# Run the Streamlit app
if __name__ == "__main__":
    main


# In[81]:


main()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




