import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif

# Sample dataset
data = {
    'age': [25, np.nan, 30, 45, 22, np.nan, 33],
    'salary': [50000, 54000, np.nan, 62000, 58000, 56000, np.nan],
    'city': ['New York', 'Los Angeles', 'Chicago', 'New York', 'Chicago', 'Los Angeles', np.nan]
}

df = pd.DataFrame(data)

# Handling Missing Values
df['age'].fillna(df['age'].mean(), inplace=True)
df['salary'].fillna(df['salary'].mean(), inplace=True)
df['city'].fillna(df['city'].mode()[0], inplace=True)

# Removing Duplicates
df.drop_duplicates(inplace=True)

# Scaling/Normalization
scaler = StandardScaler()
df[['age', 'salary']] = scaler.fit_transform(df[['age', 'salary']])

# One-Hot Encoding
df = pd.get_dummies(df, columns=['city'], drop_first=True)

# Log Transformation
df['salary'] = df['salary'].apply(lambda x: np.log(x + 1) if x > 0 else 0)

# Another sample dataset for integration
data2 = {
    'id': [1, 2, 3, 4, 5, 6, 7],
    'purchase': [250, 300, 400, 200, 150, 100, 50]
}

df2 = pd.DataFrame(data2)

# Merge datasets on index
df = pd.merge(df, df2, left_index=True, right_index=True)

# Ensure no NaN values before feature selection
df.dropna(inplace=True)

# Feature Selection
df['target'] = [0, 1, 0, 1, 0, 1, 0]  # Sample target variable
X = df.drop('target', axis=1)
y = df['target']

selector = SelectKBest(score_func=f_classif, k=2)
X_new = selector.fit_transform(X, y)

# Get selected features
selected_features = X.columns[selector.get_support()]
print("Selected Features:", selected_features)
