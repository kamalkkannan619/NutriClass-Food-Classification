# Step 3: Feature Engineering

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load cleaned dataset (after preprocessing)
df = pd.read_csv("food_data.csv")

# 1. Label Encoding for Category
encoder = LabelEncoder()
df['Category_encoded'] = encoder.fit_transform(df['Category'])

print(df[['Category', 'Category_encoded']].head())

# 2. (Optional) PCA for dimensionality reduction
from sklearn.decomposition import PCA

features = df[['Calories','Protein','Fat','Sugar']]
pca = PCA(n_components=2)
pca_features = pca.fit_transform(features)

print("Explained variance ratio:", pca.explained_variance_ratio_)
