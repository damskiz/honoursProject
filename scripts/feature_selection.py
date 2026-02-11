import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import pickle

# Get dataset name from command line
if len(sys.argv) < 2:
    print("Usage: py scripts/feature_selection.py <dataset>")
    print("Options: ddos, web, brute")
    sys.exit(1)

dataset = sys.argv[1]

# Map dataset names to file names
dataset_files = {
    'ddos': 'cleaned_friday_ddos',
    'web': 'cleaned_thursday_web',
    'brute': 'cleaned_tuesday_brute'
}

if dataset not in dataset_files:
    print(f"Error: Unknown dataset '{dataset}'")
    print("Options: ddos, web, brute")
    sys.exit(1)

# Set file paths
input_file = f'data/processed/{dataset_files[dataset]}.csv'
output_dir = 'data/processed'
results_dir = 'results'   

print("="*60)
print("FEATURE SELECTION PIPELINE")
print("="*60)

# 1. Load data
print("\n1. Loading dataset...")
df = pd.read_csv(input_file)
print(f"   Shape: {df.shape}")

# 2. Separate features and target
print("\n2. Separating features and target...")
y = df['Label']
X = df.drop('Label', axis=1)
print(f"   Features: {X.shape[1]}")
print(f"   Target classes: {y.unique()}")

# 3. Encode target (Label: DDoS/BENIGN → 1/0)
print("\n3. Encoding target variable...")
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"   Encoded classes: {np.unique(y_encoded)}")

# 4. Check if any features need encoding (shouldn't be any)
categorical_features = X.select_dtypes(include=['object']).columns
if len(categorical_features) > 0:
    print(f"\n4. One-hot encoding categorical features: {list(categorical_features)}")
    X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
else:
    print("\n4. No categorical features to encode - all numeric ✓")

print(f"   Features after encoding: {X.shape[1]}")

# 5. Correlation filtering
print("\n5. Correlation filtering (removing >0.95 correlated features)...")
correlation_matrix = X.corr().abs()
upper_triangle = correlation_matrix.where(
    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
)
high_corr_features = [
    column for column in upper_triangle.columns 
    if any(upper_triangle[column] > 0.95)
]
print(f"   Highly correlated features: {len(high_corr_features)}")
X_reduced = X.drop(columns=high_corr_features)
print(f"   Features remaining: {X_reduced.shape[1]}")

# 6. Variance filtering
print("\n6. Variance filtering (removing low variance features)...")
selector = VarianceThreshold(threshold=0.01)
X_variance = selector.fit_transform(X_reduced)
selected_features = X_reduced.columns[selector.get_support()]
print(f"   Features after variance filtering: {len(selected_features)}")

# 7. Random Forest feature importance
print("\n7. Training Random Forest for feature importance...")
X_train, X_test, y_train, y_test = train_test_split(
    X_variance, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=0)
rf.fit(X_train, y_train)
print("   Training complete ✓")

# 8. Get top features
print("\n8. Selecting top 25 features by importance...")
importances = pd.DataFrame({
    'feature': selected_features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 25 features:")
print(importances.head(25).to_string(index=False))

top_features = importances.head(25)['feature'].tolist()

# 9. Validate selection
print("\n9. Validating feature selection...")
X_final = pd.DataFrame(X_variance, columns=selected_features)[top_features]
X_train_final, X_test_final, y_train, y_test = train_test_split(
    X_final, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
rf_final = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_final.fit(X_train_final, y_train)
y_pred = rf_final.predict(X_test_final)

print("\nPerformance with selected features:")
# Get actual class names from the label encoder
target_names = le.classes_
print(classification_report(y_test, y_pred, target_names=target_names))

f1 = f1_score(y_test, y_pred, average='weighted')
print(f"\nWeighted F1-Score: {f1:.3f}")

# 10. Save everything
print("\n10. Saving results...")
with open(f'{output_dir}/selected_features_{dataset}.pkl', 'wb') as f:
    pickle.dump(top_features, f)
X_final.to_csv(f'{output_dir}/X_{dataset}_preprocessed.csv', index=False)
pd.Series(y_encoded, name='Label').to_csv(f'{output_dir}/y_{dataset}_preprocessed.csv', index=False)
importances.to_csv(f'{results_dir}/feature_importances_{dataset}.csv', index=False)

print(f"✓ Saved selected_features_{dataset}.pkl")
print(f"✓ Saved X_{dataset}_preprocessed.csv")
print(f"✓ Saved y_{dataset}_preprocessed.csv")
print(f"✓ Saved feature_importances_{dataset}.csv")

print("\n" + "="*60)
print("✅ FEATURE SELECTION COMPLETE")
print("="*60)
print(f"Started with: {X.shape[1]} features")
print(f"After correlation filtering: {X_reduced.shape[1]} features")
print(f"After variance filtering: {len(selected_features)} features")
print(f"Final selected features: {len(top_features)} features")
print(f"Performance (F1): {f1:.3f}")
print(f"Reduction: {100 * (1 - len(top_features)/X.shape[1]):.1f}%")
print("="*60)