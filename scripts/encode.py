import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load your cleaned dataset
df = pd.read_csv('data/processed/cleaned_friday_ddos.csv')  # or whatever you named it

print(f"Starting shape: {df.shape}")
print(f"\nColumn types:\n{df.dtypes.value_counts()}")

# Check for categorical features
categorical_cols = df.select_dtypes(include=['object']).columns
print(f"\nCategorical columns: {list(categorical_cols)}")

# Look at each one
for col in categorical_cols:
    print(f"\n{col}:")
    print(f"  Unique values: {df[col].nunique()}")
    print(df[col].value_counts())
