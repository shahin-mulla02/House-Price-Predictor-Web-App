import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Sample training data with more locations
data = pd.DataFrame({
    'Area': [1000, 1200, 800, 1500, 1100, 1300, 900, 1400, 1000, 1250],
    'Bedrooms': [2, 3, 2, 4, 3, 3, 2, 4, 2, 3],
    'Bathrooms': [1, 2, 1, 3, 2, 2, 1, 2, 1, 2],
    'Location': ['Mumbai', 'Delhi', 'Mumbai', 'Delhi', 'Chennai', 'Hyderabad', 'Pune', 'Bangalore', 'Kolkata', 'Chennai'],
    'Price': [5000000, 6000000, 4500000, 7500000, 5200000, 5800000, 4700000, 7200000, 5100000, 5400000]
})

# Define all possible locations
locations = ['Mumbai', 'Delhi', 'Chennai', 'Hyderabad', 'Bangalore', 'Pune', 'Kolkata']

# One-hot encode locations
for loc in locations:
    data[loc] = (data['Location'] == loc).astype(int)

# Feature matrix and target
X = data[['Area', 'Bedrooms', 'Bathrooms'] + locations]
y = data['Price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save the model
pickle.dump(model, open('model.pkl', 'wb'))
