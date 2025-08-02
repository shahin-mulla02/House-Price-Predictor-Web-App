import pickle
import numpy as np

# Load the trained model
from sklearn.linear_model import LinearRegression
import pandas as pd

# Train again or load model
df = pd.read_csv("house_data.csv")
X = df[["Area (sq ft)", "Bedrooms", "Bathrooms"]]

y = df["Price"]

model = LinearRegression()
model.fit(X, y)

# Take input
sqft = float(input("Enter square footage: "))
bed = int(input("Enter number of bedrooms: "))
bath = int(input("Enter number of bathrooms: "))

# Predict
features = np.array([[sqft, bed, bath]])
price = model.predict(features)

print(f"Predicted House Price: ₹{price[0]:,.2f}")
