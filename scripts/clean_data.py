import pandas as pd
import numpy as np

print("Loading dataset...")
df = pd.read_csv('data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
original_size = df.shape[0]

print(f"Original dataset: {original_size:,} rows")

# 1. Strip whitespace from column names
print("\n1. Fixing column names...")
df.columns = df.columns.str.strip()
print("✓ Column names cleaned")

# 2. Replace infinity values with NaN
print("\n2. Handling infinity values...")
inf_before = df.isin([np.inf, -np.inf]).sum().sum()
df = df.replace([np.inf, -np.inf], np.nan)
print(f"✓ Replaced {inf_before} infinity values with NaN")

# 3. Drop rows with missing values
print("\n3. Removing rows with missing values...")
nan_count = df.isnull().sum().sum()
df = df.dropna()
print(f"✓ Removed {original_size - df.shape[0]} rows with missing/infinity values")

# 4. Remove duplicates
print("\n4. Removing duplicate rows...")
df = df.drop_duplicates()
dup_removed = original_size - nan_count - df.shape[0]
print(f"✓ Removed {dup_removed} duplicate rows")

# 5. Verify class distribution still good
print("\n" + "="*60)
print("CLEANED DATASET SUMMARY")
print("="*60)
print(f"\nFinal shape: {df.shape}")
print(f"Rows removed: {original_size - df.shape[0]:,} ({((original_size - df.shape[0])/original_size)*100:.2f}%)")
print(f"\nClass distribution:")
print(df['Label'].value_counts())
print(f"\nPercentages:")
print(df['Label'].value_counts(normalize=True) * 100)

# 6. Save cleaned dataset
print("\n" + "="*60)
print("SAVING CLEANED DATASET")
print("="*60)
output_path = 'data/processed/cleaned_friday_ddos.csv'
df.to_csv(output_path, index=False)
print(f"✓ Saved to: {output_path}")

# 7. Save cleaning report
with open('results/cleaning_report.txt', 'w') as f:
    f.write("DATA CLEANING REPORT\n")
    f.write("="*60 + "\n\n")
    f.write(f"Original rows: {original_size:,}\n")
    f.write(f"Final rows: {df.shape[0]:,}\n")
    f.write(f"Removed: {original_size - df.shape[0]:,} ({((original_size - df.shape[0])/original_size)*100:.2f}%)\n\n")
    f.write("Steps taken:\n")
    f.write("1. Stripped whitespace from column names\n")
    f.write(f"2. Replaced {inf_before} infinity values\n")
    f.write(f"3. Dropped rows with NaN values\n")
    f.write(f"4. Removed duplicate rows\n\n")
    f.write("Final class distribution:\n")
    f.write(str(df['Label'].value_counts()))

print("✓ Cleaning report saved to: results/cleaning_report.txt")
print("\n✅ CLEANING COMPLETE!")
