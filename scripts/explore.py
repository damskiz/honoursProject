import pandas as pd

# Load just the first 1000 rows to peek
df = pd.read_csv('datasets/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', nrows=1000)

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())
print("\nLabel distribution:")
print(df[' Label'].value_counts())  # Note: might have space before 'Label'

# Check for issues
print("\nColumn data types:")
print(df.dtypes)