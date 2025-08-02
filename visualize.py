import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('house_data.csv')

# Pairplot to show relationships
sns.pairplot(df)
plt.suptitle("House Feature Relationships", y=1.02)
plt.show()

# Heatmap to show correlation
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()
