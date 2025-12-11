# Step 2: Data Preprocessing

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 1. Load dataset
df = pd.read_csv("food_data.csv")   # replace with your dataset file name
print("Dataset loaded successfully!")
print(df.head())

# 2. Check missing values
print("Missing values:\n", df.isnull().sum())

# Example: fill missing Protein values with mean
df['Protein'].fillna(df['Protein'].mean(), inplace=True)

# 3. Remove duplicates
print("Before removing duplicates:", df.shape)
df.drop_duplicates(inplace=True)
print("After removing duplicates:", df.shape)

# 4. Outlier detection (Calories example)
sns.boxplot(data=df[['Calories', 'Protein', 'Fat', 'Sugar']])
plt.show()

# 5. Feature scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['Calories', 'Protein', 'Fat', 'Sugar']])

scaled_df = pd.DataFrame(scaled_features, columns=['Calories', 'Protein', 'Fat', 'Sugar'])
print("Scaled features:\n", scaled_df.head())
