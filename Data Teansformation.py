#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
df = pd.read_csv('Employee_productivity_Dataset.csv')
print("Dataset Loaded Successfully!")
print(df.head())


# In[4]:


df['Age'] = df['Age'].fillna(df['Age'].median())
print(df['Age'])


# In[19]:


df.info


# In[8]:


df['Salary'] = df['Salary'].fillna(df['Salary'].mean())
df['Hours_Worked_Per_Week'] = df['Hours_Worked_Per_Week'].fillna(df['Hours_Worked_Per_Week'].median())
df['Performance_Score'] = df['Performance_Score'].fillna(df['Performance_Score'].mean())
print("Dataset after handling missing values")
print(df.isnull().sum())


# In[12]:


le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Department'] = le.fit_transform(df['Department'])
print("Updated Columns (Gender & Department)")
print(df[['Gender', 'Department']].head())


# In[21]:


df = pd.get_dummies(df, columns=['Work_Mode', 'Location'] ,dtype=int)
print("New columns created!")
print(df.head())


# In[16]:


from sklearn.preprocessing import MinMaxScaler
min_max = MinMaxScaler()
cols_to_normalize = ['Salary', 'Hours_Worked_Per_Week']
df[cols_to_normalize] = min_max.fit_transform(df[cols_to_normalize])
print("Normalized values (Range 0 to 1)")
print(df[cols_to_normalize].head())


# In[26]:


from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
cols_to_standardize = ['Age', 'Projects_Completed']
df[cols_to_standardize] = std_scaler.fit_transform(df[cols_to_standardize])
print(" Standardized values (Mean ≈ 0, Std ≈ 1)")
print(df[cols_to_standardize].head())


# In[28]:


salary_original = df[['Salary']] 
min_max_salary = MinMaxScaler().fit_transform(salary_original)
std_scaler_salary = StandardScaler().fit_transform(salary_original)
comparison_df = pd.DataFrame({
    'Original_Salary': salary_original.values.flatten(),
    'MinMax_Scaled': min_max_salary.flatten(),
    'Standardized_Scaled': std_scaler_salary.flatten()
})

print("Side-by-side comparison of scaling methods")
print(comparison_df.head())


# In[36]:


import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
num_cols = ['Age', 'Salary', 'Hours_Worked_Per_Week', 'Projects_Completed']
cat_cols = ['Gender', 'Department', 'Work_Mode', 'Location']

# 2. Numerical Transformer
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), 
    ('scaler', StandardScaler())
])

# 3. Categorical Transformer
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), 
    ('onehot', OneHotEncoder(handle_unknown='ignore' , spare_output=False))
])

# 4. ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])
transformed_data = preprocessor.fit_transform(df)

print("Transformed Dataset Shape", transformed_data.shape)
print("\nFirst 5 row of Transformed DataNumpy Array format():")
print(transformed_data[:5])


# In[32]:


transformed_data = preprocessor.fit_transform(df)
print("Transformed Dataset Shape", transformed_data.shape)
print("\nFirst 5 rows of Transformed Data (as an array)")
print(transformed_data[5])

