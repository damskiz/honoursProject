import pandas as pd
import numpy as np

print("Loading full dataset...")
df = pd.read_csv('data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')

print("="*60)
print("FULL DATASET EXPLORATION")
print("="*60)

print(f"\nDataset shape: {df.shape}")
print(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")

print(f"\n{'='*60}")
print("CLASS DISTRIBUTION")
print("="*60)
print(df[' Label'].value_counts())
print(f"\nPercentages:")
print(df[' Label'].value_counts(normalize=True) * 100)

print(f"\n{'='*60}")
print("DATA QUALITY ISSUES")
print("="*60)

# Missing values
missing = df.isnull().sum()
if missing.sum() > 0:
    print(f"\nMissing values found:")
    print(missing[missing > 0])
else:
    print("\nNo missing values found")

# Infinity values
inf_check = df.isin([np.inf, -np.inf]).sum()
if inf_check.sum() > 0:
    print(f"\nInfinity values found in columns:")
    print(inf_check[inf_check > 0])
else:
    print("\nNo infinity values found")

# Duplicates
dup_count = df.duplicated().sum()
print(f"\nDuplicate rows: {dup_count}")

print(f"\n{'='*60}")
print("SUMMARY SAVED")
print("="*60)

# Save distribution to file
with open('results/full_dataset_summary.txt', 'w') as f:
    f.write(f"Dataset: Friday-WorkingHours-Afternoon-DDos\n")
    f.write(f"Shape: {df.shape}\n\n")
    f.write("Class Distribution:\n")
    f.write(str(df[' Label'].value_counts()))
    f.write("\n\nPercentages:\n")
    f.write(str(df[' Label'].value_counts(normalize=True) * 100))

print("Results saved to results/full_dataset_summary.txt")
print("\n✅ Exploration complete!")
