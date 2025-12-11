# Step 6: Streamlit Dashboard

import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Title
st.title("🍎 NutriClass - Food Classification App")

# Load dataset
df = pd.read_csv("food_data.csv")

# Encode target
encoder = LabelEncoder()
df['Category_encoded'] = encoder.fit_transform(df['Category'])

# Features and target
X = df[['Calories', 'Protein', 'Fat', 'Sugar']]
y = df['Category_encoded']

# Train Random Forest
rf = RandomForestClassifier()
rf.fit(X, y)

# User input
st.sidebar.header("Enter Nutritional Values")
calories = st.sidebar.number_input("Calories", min_value=0, max_value=1000, value=100)
protein = st.sidebar.number_input("Protein (g)", min_value=0.0, max_value=100.0, value=5.0)
fat = st.sidebar.number_input("Fat (g)", min_value=0.0, max_value=100.0, value=2.0)
sugar = st.sidebar.number_input("Sugar (g)", min_value=0.0, max_value=100.0, value=10.0)

# Prediction
input_data = pd.DataFrame([[calories, protein, fat, sugar]], 
                          columns=['Calories','Protein','Fat','Sugar'])
prediction = rf.predict(input_data)
predicted_category = encoder.inverse_transform(prediction)[0]

st.subheader("Predicted Food Category:")
st.write(predicted_category)
