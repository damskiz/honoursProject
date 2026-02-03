import pandas as pd

df = pd.read_csv('data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', nrows=1000)

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist()[:10], "...")  # First 10
print("\nLabels:", df[' Label'].value_counts())